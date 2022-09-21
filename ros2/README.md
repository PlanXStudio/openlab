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
