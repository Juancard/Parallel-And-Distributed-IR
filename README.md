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

#### Python process
1) Install python  modules: nltk y numpy
2) Rename file `config.ini.example` to `config.ini`
3) Edit values of properties in file `config.ini`
4) Run process: `python sockets.py [-dv]`

#### Broker


#### IR server
1) Add libraries needed: 
-- Json: https://mvnrepository.com/artifact/org.json/json/20140107
-- Jsch: http://www.java2s.com/Code/Jar/j/Downloadjsch0142jar.htm
-- Guava: http://central.maven.org/maven2/com/google/guava/guava/16.0.1/guava-16.0.1.jar
-- Ini4j: https://mvnrepository.com/artifact/org.ini4j/ini4j/0.5.1

2) Rename file `config.properties.example` to `config.properties`
3) Edit values of properties in `config.properties`
4) Run View.InitServer
 

