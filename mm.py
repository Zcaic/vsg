import concurrent.futures
import time

def _target(number):
    for i in range(int(1e8)):
        i=i+1
    return number

if __name__=="__main__":
    t1=time.time()
    for i in range(3):
        print(_target(i))
    t2=time.time()
    print(f"exec in {t2-t1} seconds")

    t3=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        p1=executor.submit(_target,2)
        p2=executor.submit(_target,3)
        p3=executor.submit(_target,4)
        futures=[p1,p2,p3]
        r=[]
        concurrent.futures.wait(futures)
        r.append(p1.result())
        r.append(p2.result())
        r.append(p3.result())
        
    print(r)
    t4=time.time()
    print(f"exec in {t4-t3} seconds")