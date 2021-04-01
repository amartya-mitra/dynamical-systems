                # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'previous_iterate' not in param_state:
                #         prev = param_state['previous_iterate'] = p.data.detach()
                #     else:
                #         # import ipdb; ipdb.set_trace()
                #         prev = param_state['previous_iterate']

                # param_state['previous_iterate'] = p.data
                # p.data = p.data + momentum * (p.data - prev.data) - group['lr'] * d_p


                               # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'v' not in param_state:
                #         v = param_state['v'] = torch.zeros_like(p.data)
                #     else:
                #         v = param_state['v']