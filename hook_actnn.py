from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook

import actnn
import torch

@HOOKS.register_module()
class ActNNHook(Hook):

    def __init__(self, default_bit=4):
        self.default_bit = default_bit

    def after_run(self, runner):
        del runner.controller

    def before_run(self, runner):
        controller = actnn.controller.Controller(
            default_bit=self.default_bit)
        runner.controller = controller
        controller.install_hook()

    # def before_train_epoch(self, runner):
    #     model = (runner.model.module if is_module_wrapper(
    #         runner.model) else runner.model)
    #     runner.controller.unrelated_tensors = set()
    #     runner.controller.filter_tensors(model.named_parameters())

    def after_train_iter(self, runner):
        runner.controller.iterate()