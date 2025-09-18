



def dummy_controller(queue):
    while True:
        msg = queue.get()
        print(f"UART data:{msg}")
        if msg == "STOP":
            continue
        # просто чекаємо, не робимо нічого



# тут повинен бути  код який передає по UART  порту команди  на esp32, педати потрібно мсасив даних  типу 
#  command = ["u", "stop":false] це потрібно обновляти  time( 0.3 )   якщо u =22 то  передати  command = ["stop": true]
# но потрібно щоб воно працювала в 2 осях  Х У одночсно   command =- ["u","l", "stop":false] якщо  u = 22  то u не передаєтся
# 