import time 
start_time = time.time()
for i in range(1000000000):
    test = 3 + 5
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")