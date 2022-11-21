"""
@PROJECT: DySAT_pytorch - mylogging.py
@IDE: PyCharm
@DATE: 2022/11/20 下午5:01
@AUTHOR: lxx
"""
import os.path
import time
from collections import defaultdict

from torch import nn
from torch.utils.tensorboard import SummaryWriter


def mk_writer_dir(path: str) -> str:
    dir_date = str(time.gmtime().tm_year) + '-' + str(time.gmtime().tm_mon) + '-' + str(time.gmtime().tm_mday) + '-'
    dir_id = 0
    dir_path = path + dir_date + str(dir_id)
    while os.path.isdir(dir_path):
        dir_id += 1
        dir_path = path + dir_date + str(dir_id)
    return dir_path


class LoggingWriter:
    def __init__(self, path: str = 'log/'):
        self.writer_path = mk_writer_dir(path)
        self.writer = SummaryWriter(log_dir=self.writer_path)
        self.epoch_loss = defaultdict(list)

    def summary_loss(self, loss: float, epoch: int, global_step: int):
        """
        记录完整训练流程中的loss变化
        Args:
            loss:
            epoch:
            global_step: 当前step,计算 = epoch * len(Dataloader) + step
        """
        self.writer.add_scalar('The summary of loss', scalar_value=loss, global_step=global_step)
        self.epoch_loss[f'epoch-{epoch}'].append(loss)

    def summary_global_loss(self):
        """
        记录每epoch中训练的loss变化,在一张图中展示
        展示的数据源于summary_loss中的记录
        """
        if len(self.epoch_loss) == 0:
            return False
        max_len = max([len(el) for el in self.epoch_loss.values()])
        print(self.epoch_loss)
        print('max_len: ', max_len)
        for i in range(max_len):
            self.writer.add_scalars('The summary of global loss',
                                    tag_scalar_dict={k: -1 if len(self.epoch_loss[k]) <= i else self.epoch_loss[k][i]
                                                     for k in self.epoch_loss.keys()},
                                    global_step=i)

    def summary_model(self, model: nn.Module, input_data: any):
        self.writer.add_graph(model=model, input_to_model=input_data.values, verbose=False)

    def summary_action(self):
        port = 5333
        title_name = 'test summary'
        os.system(f'tensorboard --logdir {self.writer_path} --port {port} --window_title "{title_name}"')

    def summary_close(self):
        self.writer.close()


if __name__ == '__main__':
    writer = SummaryWriter()
    a = [1, 2, 3, 4]
    b = [6, 3, 2, 6]
    c = [3, 6, 8, 1]
    # for i in range(4):
    #     writer.add_scalar('The su of loss', scalar_value=i, global_step=i)
        # writer.add_scalars('test', {'xsinx': a[i],
        #                                 'xcosx': b[i],
        #                                 'tanx': c[i]}, i)
    writer.close()
