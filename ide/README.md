# VSCode 

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

# PC to Soda (arm64)

## Connection
PC(Windows 11) <-> Ethernet USB((192.168.101.120) <------SSH-----> Ethernet(192.168.101.101) <-> Soda OS

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

rqt running [Soda] <--------**nomachine**------> [Windows 11] Screen & Keyboard/Mouse Event

## [option] Wi-Fi AP connect
```sh
nmcli device wifi list
sudo nmcli device wifi connect <ssid> password <passwd>
sudo nmcli connect delete <ssid>
```

## VSCode portable (PC)

a. VSCode download (Last Version)
- origin
  - https://code.visualstudio.com/Download#  > Windows > .zip > 64 bit
- backup
  - https://fossies.org/windows/misc/ > VSCode-win32-x64-<x.y.z>.zip

b. unzip VSCode-win32-x64-<x.y.z>.zip 
- C:\VScode

c. mkdir data folder into C:\VSCode
- C:\VScode\data

d. run VSCode
- C:\VSCode\Code.exe

e. install extension
  - Remote Development

**Pre-install Link**
[VSCode_1.72.0](https://koreaoffice-my.sharepoint.com/:u:/g/personal/devcamp_korea_edu/EfKKdu6ECGdGoa6KfYMvl5cB7t8Svh6ccIeQSJoVBj4W2A?e=uSRRSK)
-COMMIT_ID: 64bbfbf67ada9953918d72e1df2f4d8e537d340e

## Remote development environment (WSL Linux or Soda)
a. Connection to WSL Linux (WSL) or Soda (SSH)
b. automatic install vscode-server  
- *download server*
  - X64
    - last: https://update.code.visualstudio.com/latest/server-linux-x64/stable
  - ARM64
    - last: https://update.code.visualstudio.com/latest/server-linux-arm64/stable
  - COMMIT ID
    -latest to commit:$COMMIT_ID (ex: https://update.code.visualstudio.com/commit:$COMMIT_ID...)

**menual install** (network issue etc)  
a. Check commit-id
   - Run CMD or PowerShell
```sh
cd C:\VSCode\bin
./code --version | wsl sed -n '2p'
```
>> ex) 64bbfbf67ada9953918d72e1df2f4d8e537d340e

b. Download vscode-server  
   - Run wsl linux  

**WSL Linux**
```sh
wget -q --show-progress --progress=bar:force --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -c -t 0 -O vscode-server.tar.gz https://update.code.visualstudio.com/latest/server-linux-x64/stable
```

**Soda (arm64)**
```sh
wget -q --show-progress --progress=bar:force --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -c -t 0 -O vscode-server.tar.gz https://update.code.visualstudio.com/latest/server-linux-arm64/stable
```

c. Extract vscode-server (WSL Linux or Soda)
```sh
mkdir -p ~/.vscode-server/bin/<commit-id>
mv vscode-server.tar.gz ~/.vscode-server/bin/<commit-id>
tar -xvzf vscode-server.tar.gz --strip-components 1
rm -rf vscode-server.tar.gz
```

**Pre-install Link**
[Soda(arm64)](https://koreaoffice-my.sharepoint.com/:u:/g/personal/devcamp_korea_edu/EaqBzKL12IxKvgcmJS_baZQBEcg0as0huHlfyibw4AtpOw?e=cUZKcr)
-for VSCode_1.72.0

<br/>

