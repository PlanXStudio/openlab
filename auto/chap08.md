
A. 같은 패키지에서 구독과 발행
- 패키지 (ament_python)
  - ex_topic
  - 노드
    - ex_pub
    - ex_sub

```python:hello.py
ros2 pkg create --build-type ament_python --node-name ex_pub ex_topic
touch ex_topic/ex_topic/ex_sub.py
```


B. 개별 패키지에서 구독과 발행
- 패키지 (ament_python)
  - ex_pub
  - 노드
    - ex_pub_alone
- 패키지 (ament_python)
  - ex_sub
  - 노드
    - ex_sub_alone

C. 사용자 인터페이스
- 패키지 (amnet_cmake)
  - ex_interface
  - 
