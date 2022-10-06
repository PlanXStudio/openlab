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

**Pre-install Link**
[VSCode_1.71.2](https://koreaoffice-my.sharepoint.com/:u:/g/personal/devcamp_korea_edu/EV9A0jS501RDg65GcAWFe6gBnlVTXhIE97vqY8COArk_yg?e=GPC6jr)
-COMMIT_ID: 74b1f979648cc44d385a2286793c226e611f59e7

## vscode-server (WSL Linux or Soda)
1. VSCode(Windows 11) connection to WSL Linux or Soda
2. automatic install vscode-server  
- *download server*
  - X64
    - last: https://update.code.visualstudio.com/latest/server-linux-x64/stable
    - CONNIT_ID use: https://update.code.visualstudio.com/commit:$COMMIT_ID/server-linux-x64/stable

  - ARM64
    - last: https://update.code.visualstudio.com/latest/server-linux-arm64/stable
    - CONNIT_ID use: https://update.code.visualstudio.com/commit:$COMMIT_ID/server-linux-arm64/stable

**menual install** (network issue etc)  
1. Terminal connection to WSL Linux or Soda
2. Check commit-id
```sh
ls ~/.vscode-server/bin/
```
*commit-id*

3. Set commit-id
```sh
COMMIT_ID=$(ls ~/.vscode-server/bin)
```
*or*
```sh
COMMIT_ID=$(ls -tral -1 ~/.vscode-server/bin | sed -n '3p' | rev | cut -d' ' -f1 | rev)
```

```sh
cd ~/.vscode-server/bin/$COMMIT_ID
```
4. Download vscode-server
**WSL Linux(x64)**
```sh
wget -q --show-progress --progress=bar:force -retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -c -t 0 -O vscode-server.tar.gz https://update.code.visualstudio.com/commit:$COMMIT_ID/server-linux-x64/stable
```

**Soda(arm64)**
```sh
wget -q --show-progress --progress=bar:force -retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -c -t 0 -O vscode-server.tar.gz https://update.code.visualstudio.com/commit:$COMMIT_ID/server-linux-arm64/stable
```

**Pre-install Link for Soda(arm64)**
[vscode-server.tar.gz](https://koreaoffice-my.sharepoint.com/:u:/g/personal/devcamp_korea_edu/EaqBzKL12IxKvgcmJS_baZQBEcg0as0huHlfyibw4AtpOw?e=WtzrXz)
-for VSCode_1.71.2 (COMMIT_ID: 74b1f979648cc44d385a2286793c226e611f59e7)

5. Extract
```sh
tar -xvzf vscode-server.tar.gz --strip-components 1
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

# PC to Soda

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
