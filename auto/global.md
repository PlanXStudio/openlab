a. 전역 작업 공간 설정. 이후 모든 패키지는 이 작업 공간 사용
```sh
mkdir -p ~/Project/auto/src
```

b. colcon_cd 작업 공간 루트 설정

```sh
export _colcon_cd_root=~/Project/auto
```

c. argcomplete for ros2 & colcon
```sh
source /opt/ros/foxy/setup.zsh
export PATH=/mnt/c/VSCode/bin/:$PATH

source /usr/share/colcon_cd/function/colcon_cd.sh
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.zsh

eval "$(register-python-argcomplete3 ros2)"
eval "$(register-python-argcomplete3 colcon)"
```
---

A. 패키지 생성

```sh
colcon_cd
cd src
ros2 pkg create --build-type ament_python --node-name <node name> <package name>
```
  
B-1 패키지 구현
```sh
src\<package name>\<pacakge name>\<node name>.py
```
  
B-2 package.xml 설정
```sh
dependency...
```
  
B-3 setup.py 설정
```sh
'console_scripts': [
    '<executable name> = <package name>.<node name>:main',
    ...
],
```
  
C. 패키지 빌드
```sh
colcon_cd
colcon build packages-select <package name>
```
  
D. 실행 공간 소싱
```sh
colcon_cd
source ./install/setup.zsh
```

E. 패키지 실행
```sh
ros2 run <package name> <executable name>
```
---
