# parallel-and-distributed-IR

## Overview

_parallel-and-distributed-IR_ es un sistema distribuido y paralelo de recuperación de información en el cual existe una colección de documentos de texto libre o corpus y usuarios que realizan consultas o queries en busca de satisfacer una necesidad de información. 

Este sistema propone aplicar técnicas de paralelización al momento de la resolución de queries del usuario de manera tal de optimizar los tiempos de respuesta. De igual manera, el sistema constará de una serie de dispositivos o nodos interconectados que garanticen una mayor confiabilidad, seguridad y disponibilidad de los servicios y datos administrados.

Propuesta completa [aquí][1]

## Architecture

![arquitectura de sistema](http://i.imgur.com/AzY7S5x.png)

[1]: https://drive.google.com/open?id=1-hrXE356gGyHKITsZKsmqjxnsO6fwFIvwg2OK6spN-8

## Installation

### IR server

#### IR server

##### Python process
1) Install python  modules: nltk y numpy
2) Rename file `config.ini.example` to `config.ini`
3) Edit values of properties in file `config.ini`
4) Run process: 
```bash
python sockets.py [-dv]
```

##### Java process
1) Add libraries needed: 
- Json: https://mvnrepository.com/artifact/org.json/json/20140107
- Jsch: http://www.java2s.com/Code/Jar/j/Downloadjsch0142jar.htm
- Guava: http://central.maven.org/maven2/com/google/guava/guava/16.0.1/guava-16.0.1.jar
- Ini4j: https://mvnrepository.com/artifact/org.ini4j/ini4j/0.5.1

2) Rename file `config.properties.example` to `config.properties`
3) Edit values of properties in `config.properties`
4) Open ssh tunnel (You'll be asked to enter user and pass in the remote gpu server)
```ssh -L 3491:localhost:3491 your_user@170.210.103.21```
5) Run View.InitServer


 
#### Broker
1) Rename file `IR_servers.cfg.example` to `IR_servers.cfg`
2) To add more servers, add their address to `IR_servers.cfg`
3) Rename file `config.properties.example` to `config.properties`
4) Edit values of properties in `config.properties`
5) Run View.InitBroker
 
#### IR client
1) Rename file `config.properties.example` to `config.properties`
2) Edit values of properties in `config.properties`
3) Run View.InitClient
