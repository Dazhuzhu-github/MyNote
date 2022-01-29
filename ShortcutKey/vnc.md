1 安装vnc viewer
https://www.realvnc.com/en/connect/download/viewer/windows/

2 ssh到服务器

3 配置xstarpup

```
#!/bin/sh

xrdb $HOME/.Xresources
xsetroot -solid grey
#x-terminal-emulator -geometry 80x24+10+10 -ls -title "$VNCDESKTOP Desktop" &
#x-window-manager &
# Fix to make GNOME work
export XKL_XMODMAP_DISABLE=1
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
mate-session&
mate-terminal&
/etc/X11/Xsession
```

4 启动vncserver

```
vncserver -geometry 1920x1080
```

会有类似

```
New 'X' desktop is AI0:6

Starting applications specified in /home/zhmc/.vnc/xstartup
Log file is /home/zhmc/.vnc/AI0:6.log
```

5 vncviewer连接

```
ip + : + 编号
ex：111.22.11 :6
```

6 删除server,冒号后是自己的编号

```
vncserver -kill :6
```

7 顯示所有vnc
```
pgrep vnc
```

