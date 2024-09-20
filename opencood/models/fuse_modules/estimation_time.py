
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision.models as models

class ModelAnalyzer:
    def __init__(self, model,flag=549, time_logger=False):
        self.model = model
        self.flag = flag
        self.time_logger= time_logger

    def analyze(self,batch_data,iter, start_list, end_list):

        if iter==self.flag:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                        record_shapes=True,
                        profile_memory=True,  # 开启内存分析
                        with_flops=True       # 开启FLOPs分析
                        ) as prof:
                with record_function("model_inference"):
                    self.model(batch_data[0],batch_data[1])

            self.profiler = prof

            # Print total model parameters in millions (M)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"The model parameters: {total_params / 1e6:.8f}M")

            # # Print total FLOPs in GFLOPs
            total_flops = sum(item.flops for item in self.profiler.key_averages())
            print(f"The model FLOPs: {total_flops / 1e9:.8f} GFLOPs")

            # Print total GPU memory usage in MB
            total_gpu_memory = sum(item.cuda_memory_usage for item in self.profiler.key_averages())
            print(f"The model GPU Memory: {total_gpu_memory / (1024 ** 2):.8f} MB")

            # Print total CPU memory usage in MB
            total_cpu_memory = sum(item.cpu_memory_usage for item in self.profiler.key_averages())
            print(f"The model CPU Memory: {total_cpu_memory / (1024 ** 2):.8f} MB")

            # self.profiler.export_chrome_trace("trace.json")


            if self.time_logger:
                num_runs = len(start_list) 
                total_time = end_list[-1] - start_list[0]
                average_fps = num_runs / total_time
                print('\n The model total time for {} runs: {:.8f} seconds'.format(num_runs, total_time))
                print('The model Average FPS: {:.8f}'.format(average_fps))
                print(' ')


class ModelAnalyzer1:
    def __init__(self, model,flag=549, time_logger=False):
        self.model = model
        self.flag = flag
        self.time_logger= time_logger

    def analyze(self,batch_data,iter, start_list, end_list):

        if iter==self.flag:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                        record_shapes=True,
                        profile_memory=True,  # 开启内存分析
                        with_flops=True       # 开启FLOPs分析
                        ) as prof:
                with record_function("model_inference"):
                    self.model(batch_data[0],batch_data[1],batch_data[2])

            self.profiler = prof

            # Print total model parameters in millions (M)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"The model parameters: {total_params / 1e6:.8f}M")

            # # Print total FLOPs in GFLOPs
            total_flops = sum(item.flops for item in self.profiler.key_averages())
            print(f"The model FLOPs: {total_flops / 1e9:.8f} GFLOPs")

            # Print total GPU memory usage in MB
            total_gpu_memory = max(item.cuda_memory_usage for item in self.profiler.key_averages())
            print(f"The model GPU Memory: {total_gpu_memory / (1024 ** 2):.8f} MB")

            # Print total CPU memory usage in MB
            total_cpu_memory = max(item.cpu_memory_usage for item in self.profiler.key_averages())
            print(f"The model CPU Memory: {total_cpu_memory / (1024 ** 2):.8f} MB")

            # self.profiler.export_chrome_trace("trace.json")


            if self.time_logger:
                num_runs = len(start_list) 
                total_time = end_list[-1] - start_list[0]
                average_fps = num_runs / total_time
                print('\n The model total time for {} runs: {:.8f} seconds'.format(num_runs, total_time))
                print('The model Average FPS: {:.8f}'.format(average_fps))
                print(' ')


def analyze(model,batch_data,num_runs,total_time):

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                    record_shapes=True,
                    profile_memory=True,  # 开启内存分析
                    with_flops=True       # 开启FLOPs分析
                    ) as prof:
            with record_function("model_inference"):
                model(batch_data)
        
        profiler = prof

        # Print total model parameters in millions (M)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"The Final total parameters: {total_params / 1e6:.8f}M")
        print('')

        # Print total FLOPs in GFLOPs
        total_flops = sum(item.flops for item in profiler.key_averages())
        print(f"The Final total FLOPs: {total_flops / 1e9:.8f} GFLOPs")
        print('')

        # Print total GPU memory usage in MB
        # total_gpu_memory = sum(item.cuda_memory_usage for item in profiler.key_averages())
        total_gpu_memory = max(item.cuda_memory_usage for item in profiler.key_averages())
        print(f"The Final Peak GPU  Memory: {total_gpu_memory / (1024 ** 2):.8f} MB")
        print('')

        # Print total CPU memory usage in MB
        total_cpu_memory = max(item.cpu_memory_usage for item in profiler.key_averages())
        print(f"The Final Peak CPU Memory: {total_cpu_memory / (1024 ** 2):.8f} MB")

        average_fps = num_runs / total_time
        print(f"Final Total time for {num_runs} runs: {total_time:.8f} seconds")
        print('')
        print(f"Final Average FPS: {average_fps:.8f}")
        print('')

        # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

        # 获取并打印峰值显存使用量
        max_memory_allocated = torch.cuda.max_memory_allocated()
        max_memory_reserved = torch.cuda.max_memory_reserved()

        print(f"最大显存分配量: {max_memory_allocated / (1024 ** 2):.8f} MB")
        print(f"最大显存保留量: {max_memory_reserved / (1024 ** 2):.8f} MB")








# 使用示例
# model = models.resnet50(pretrained=True)
# analyzer = ModelAnalyzer(model)
# analyzer.analyze()

