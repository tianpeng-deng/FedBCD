# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
import torch.nn as nn
from collections import defaultdict

class clientBCD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        if id == 0: 
            self.model = copy.deepcopy(args.model_3D)
            self.batch_size = 8
        else: 
            self.model = copy.deepcopy(args.model)
        
        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        
        self.trainable_dict = {k: v for k, v in self.model.named_parameters() if v.requires_grad}
        
        test = 1
        # ## freeze some parameters
        # for name, param in args.model.named_parameters():
        #     if 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name and 'fc_cls' not in name:
        #         param.requires_grad = False
        #     else:
        #         param.requires_grad = True
        
    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        dt_size = len(trainloader.dataset)
        all_step = (dt_size-1)//trainloader.batch_size
        self.grad_weight = defaultdict(lambda: 0.0)
        for epoch in range(max_local_epochs):
            print(f"{self.id} Local {epoch}")
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                
                loss.backward()
                for layer_name, param in self.trainable_dict.items():
                    if 'temporal_embedding' in layer_name:
                        continue
                    if 'bias' in layer_name or layer_name == 'ln_post.weight':
                        self.grad_weight[layer_name] += (param.grad**2)
                    else:
                        self.grad_weight[layer_name] += param.grad @ param.grad.T
                        
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        for layer_name, param in self.trainable_dict.items():
            self.grad_weight[layer_name] /= (all_step * max_local_epochs)
                
        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
            
            
    def set_aggregated_layers(self, aggregated_weight):
        local_model_state_dict = self.model.state_dict()
        for key in aggregated_weight.keys():
            local_model_state_dict[key] = aggregated_weight[key]

        self.model.load_state_dict(local_model_state_dict, strict=True)