using CUDA

if Sys.iswindows()
    function hello_world()
        @cuprintf("Greetings from block %lld, thread %lld!\n", Int64(blockIdx().x), Int64(threadIdx().x))
        return
    end
else
    function hello_world()
       @cuprintf("Greetings from block %ld, thread %ld!\n", Int64(blockIdx().x), Int64(threadIdx().x))
       return
   end
end
@cuda blocks=20000 threads=16 hello_world()
synchronize()
