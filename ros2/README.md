# PC to AutoCar3

## Connection
PC(Windows 11) <-> Ethernet USB((192.168.101.120) <------SSH-----> Ethernet(192.168.101.101) <-> Autocar3(Soda OS)

*Soda OS*
- IP: 192.168.101.101
- ID: soda
- PW: soda

## Remote desktop install
nomachine for Windows (https://www.nomachine.com/download/download&id=8)  

## Remote SSH connection (PC to AutoCar3)

```sh
ssh soda@192.168.101.101
```

## AutoCar3 Wi-Fi AP connect
```sh
nmcli device wifi list
sudo nmcli device wifi connect <ssid> password <passwd>
sudo nmcli connect delete <ssid>
```

## Run Remote GUI App
```sh
export DISPLAY=:0
rqt
```

rqt running [AutoCar3] <--------nomachine------> [Windows 11] Screen & Keyboard/Mouse Event
