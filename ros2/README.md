# Soda OS

## Ethernet USB (192.168.101.120)
IP: 192.168.101.101
ID: soda
PW: soda

## Remote desktop
Install nomachine for Windows (https://www.nomachine.com/download/download&id=8)  

## Remote SSH

### connection
```sh
ssh soda@192.168.101.101
```

### ap connect
```sh
nmcli device wifi list
sudo nmcli device wifi connect <ssid> password <passwd>
sudo nmcli connect delete <ssid>
```

### run gui app
```sh
export DISPLAY=:0
rqt
```
