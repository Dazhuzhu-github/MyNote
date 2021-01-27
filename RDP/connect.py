import os, sys, time, chardet
import subprocess
import requests
import hashlib

def is_net_ok(ping_target):
    #null = open(os.devnull, 'w');
    #res = subprocess.call('ping 8.8.8.8', shell = True, stdout = null, stderr = null);
    
    p = subprocess.Popen("ping " + ping_target, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE);
    (stdoutput, erroutput) = p.communicate();
    encoding = chardet.detect(stdoutput)['encoding'];
    output = stdoutput.decode(encoding);
    retcode = p.returncode;
    res = ("ms TTL=" not in output);
    
    if res:
        print('Ping failed.');
        return False;
    else:
        print('Ping success.');
        return True;

def wlan_connect(name, interface):
    null = open(os.devnull, 'w');
    res = subprocess.call('netsh wlan connect name="' + name + '" interface="' + interface + '"', shell = True, stdout = null, stderr = null);
        
    if res:
        print('Connect wlan-Tsinghua failed.');
        return False;
    else:
        print('Connect wlan-Tsinghua success.');
        return True;
    
def login(username, password):
    data = {
        'action' : 'login',
        'username' : username,
        'password' : '{MD5_HEX}' + hashlib.md5(password.encode()).hexdigest(),
        'ac_id' : '1'
    }; # 不用自己进行urlencode

    headers = {
        'Host': 'net.tsinghua.edu.cn',
        'Origin': 'http://net.tsinghua.edu.cn',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Referer': 'http://net.tsinghua.edu.cn/wired/',
    };
 
    try:
        response = requests.post('http://net.tsinghua.edu.cn/do_login.php', data=data, headers=headers, timeout=10);
        print(response.text);
    except:
        print("Unfortunitely -- An error happended on requests.post()")

if __name__ == '__main__':
    
    username = "zhengmc19";
    password = "zheng7601";
    interface = "WLAN" # "Wireless Network Connection"
    name = "eesast904"
    
    if len(sys.argv) > 3:
        interface = sys.argv[3]

    while True:
        if not is_net_ok("baidu.com"):
            print("\n");
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())));
            print('The network is disconnected.');
            if wlan_connect(name, interface):
                time.sleep(5); # Win10连接wlan之后会立即自动弹出登录页面，造成"getaddrinfo failed"
                login(username, password);
        else:
            time.sleep(1);
        



