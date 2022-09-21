# VSCode 

## VSCode portable (PC)

1. VSCode download
- https://fossies.org/windows/misc/ 
  - VSCode-win32-x64-<version>.zip

2. unzip VSCode-win32-x64-<version>.zip 
- C:\VScode

3. mkdir data folder into C:\VSCode
- C:\VScode\data

4. run VSCode
- C:\VSCode\Code.exe

5. install extension
  - Remote Development

## vscode-server (WSL Linux or AutoCar3)
1. VSCode(Windows 11) connection to WSL Linux or AutoCar3  
2. automatic install vscode-server  
- *download server*
  - X64: *https://update.code.visualstudio.com/latest/server-linux-x64/stable*
    - CONNIT_ID: https://update.code.visualstudio.com/commit:${COMMIT_ID}/server-linux-x64/stable

  - ARM64: *https://update.code.visualstudio.com/latest/server-linux-arm64/stable*  
    - CONNIT_ID: https://update.code.visualstudio.com/commit:${COMMIT_ID}/server-linux-arm64/stable
  
**menual install** (network issue etc)  
1. Terminal connection to WSL Linux or AutoCar3 
2. Check commit-id
```sh
ls ~/.vscode-server/bin/
```
*commit-id*

3. Set commit-id
```sh
COMMIT_ID=commit-id
cd ~/.vscode-server/bin/COMMIT_ID
```

4-1. download vscode-server for WSL Linux(x64)
```sh
wget --progress=bar --tries=1 --connect-timeout=3 --dns-timeout=3 -O vscode-server.tar.gz https://update.code.visualstudio.com/commit:${COMMIT_ID}/server-linux-x64/stable
```

4-2. download vscode-server for AutoCar3(arm64)
```sh
wget --progress=bar --tries=1 --connect-timeout=3 --dns-timeout=3 -O vscode-server.tar.gz https://update.code.visualstudio.com/commit:${COMMIT_ID}/server-linux-arm64/stable
```

5. Extract
```sh
tar -xvzf vscode-server.tar.gz --strip-components 1
```

### wget option
take over
```sh
-c
```
Show status bar only: 
```sh
-q --show-progress --progress=bar:force
```
  
<br/>

# PC to WSL Linux

## Connection
PC(Windows 11) <-> vEthernet WSL <------VM-----> vEthernet(eth0) <-> Ubuntu

```sh
wsl
cd ~
```

## Run Remote GUI App
```sh
rqt
```

rqt running [WSL Linux] <--------VM------> [Windows 11] Screen & Keyboard/Mouse Event

<br/>  

# PC to AutoCar3

## Connection
PC(Windows 11) <-> Ethernet USB((192.168.101.120) <------SSH-----> Ethernet(192.168.101.101) <-> AutoCar3(Soda OS)

*Soda OS*
- IP: 192.168.101.101
- ID: soda
- PW: soda

## Remote Desktop
1. install nomachine for Windows (https://www.nomachine.com/download/download&id=8)  
2. **run nomachine > 192.168.101.101**


## Remote SSH connection 

```sh
ssh soda@192.168.101.101
```

## Run Remote GUI App
```sh
export DISPLAY=:0
rqt
```

rqt running [AutoCar3] <--------**nomachine**------> [Windows 11] Screen & Keyboard/Mouse Event

## [option] AutoCar3 Wi-Fi AP connect
```sh
nmcli device wifi list
sudo nmcli device wifi connect <ssid> password <passwd>
sudo nmcli connect delete <ssid>
```
