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

import time
from flcore.clients.clientbcd import clientBCD
from flcore.servers.serverbase import Server
from threading import Thread
import copy
import torch
import random
from collections import defaultdict
import math
import pickle
import os

class FedBCD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.global_model = copy.deepcopy(args.model)
        self.global_model_3D = copy.deepcopy(args.model_3D) if hasattr(args, 'model_3D') else None

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientBCD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        
        self.all_epoch_fims = []
        global scale
        scale = args.scale
        self.scale = args.scale

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            if i == 0:
                self.send_models()
            else:
                self.send_aggregated_layers()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                
                # self.save_global_model(i) 
                if i == self.global_rounds:
                    self.save_global_model(i)

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # self.save_global_model()

            
    def save_global_model(self, epoch_id):
        model_path = os.path.join("models", self.dataset, self.fold, self.algorithm)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        global_model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        # torch.save(self.global_model, global_model_path)
        
        for client in self.clients:
            local_model_path = os.path.join(model_path, self.algorithm + f'Epoch_{epoch_id}_Client_{client.id}_scale_{self.scale}' + '.pt')
            torch.save(client.model, local_model_path)
            
    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            if client.id != 0:
                client.set_parameters(self.global_model)
            elif client.id == 0:
                client.set_parameters(self.global_model_3D)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    
    def send_aggregated_layers(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            client.set_aggregated_layers(self.agg_global_model[client.id])
            
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    
    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = self.selected_clients
        
        self.uploaded_ids = []
        self.uploaded_weights = defaultdict(list)  # layerwise weights
        self.uploaded_models = {}   # joint trainable layers
        tot_samples = 0
        client_fisher_trace_dict = defaultdict(list)
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_models[client.id] = get_layer_param(client.model.state_dict())
                
                client_fisher_trace_dict = get_layer_fisher(client_fisher_trace_dict, client.grad_weight)
        self.all_epoch_fims.append(client_fisher_trace_dict)    # Record FIM
        self.uploaded_weights = get_layer_weight(client_fisher_trace_dict)
        
    def aggregate_parameters(self):
        self.agg_global_model = copy.deepcopy(self.uploaded_models)
        all_base = dict()
        for key, value in self.uploaded_models[0].items():
            all_base[key] = torch.zeros_like(value)
            
        for key in all_base.keys():
            name_prefix = '.'.join(key.split('.')[:-1])
            for client_id, state_dict in self.uploaded_models.items():
                tmp_weight = state_dict[key]
                all_base[key] += self.uploaded_weights[name_prefix][client_id] * tmp_weight
        
        for client_name, _ in self.agg_global_model.items():
            self.agg_global_model[client_name] = copy.deepcopy(all_base)
        

def get_layer_param(state_dict):
    param_need_to_aggregate = dict()
    for layer_name, param in state_dict.items():
        if 'MLP_Adapter.D_fc2' in layer_name:
            continue
        if 'S_Adapter' in layer_name or 'MLP_Adapter' in layer_name or 'cls_head' in layer_name or 'head' in layer_name: 
            param_need_to_aggregate[layer_name] = param
    
    return param_need_to_aggregate

def get_layer_fisher(client_fisher_trace_dict, grad_weight):
    for layer_name, layer_fisher_matrix in grad_weight.items():
        if 'weight' in layer_name and 'ln_post' not in layer_name:
            name_prefix = '.'.join(layer_name.split('.')[:-1])
            client_fisher_trace_dict[name_prefix].append(torch.trace(layer_fisher_matrix).data)
    
    return client_fisher_trace_dict

def get_layer_weight(client_fisher_trace_dict):
    client_layer_weight = defaultdict(list)
    for key, value in client_fisher_trace_dict.items():
        # # if 'MLP_Adapter.D_fc2' in key:
        # #     client_layer_weight[key] = softmax([i/5 for i in value])
        # # else:
        # #     client_layer_weight[key] = softmax(value)
        # client_layer_weight[key] = softmax(value)
        
        client_layer_weight[key] = softmax([i/scale for i in value])
    return client_layer_weight

def softmax(x):
    e_x = [math.exp(-xi) for xi in x]
    sum_e_x = sum(e_x)
    return [ei / sum_e_x for ei in e_x]