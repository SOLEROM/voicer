# pipewire

> pipewire --version
pipewire
Compiled with libpipewire 0.3.65
Linked with libpipewire 0.3.65



rock@rock-5b:~$ systemctl --user status pipewire
● pipewire.service - PipeWire Multimedia Service
     Loaded: loaded (/usr/lib/systemd/user/pipewire.service; enabled; preset: enabled)
     Active: active (running) since Thu 2025-06-12 11:01:32 UTC; 1s ago
TriggeredBy: ● pipewire.socket
   Main PID: 1965 (pipewire)
      Tasks: 2 (limit: 9192)
     Memory: 3.4M
        CPU: 75ms
     CGroup: /user.slice/user-1001.slice/user@1001.service/session.slice/pipewire.service
             └─1965 /usr/bin/pipewire

Jun 12 11:01:32 rock-5b systemd[1957]: Started pipewire.service - PipeWire Multimedia Service.
Jun 12 11:01:32 rock-5b pipewire[1965]: mod.rt: Can't find org.freedesktop.portal.Desktop. Is xdg-desktop-portal running?
Jun 12 11:01:32 rock-5b pipewire[1965]: mod.rt: found session bus but no portal
Jun 12 11:01:32 rock-5b pipewire[1965]: mod.rt: RTKit error: org.freedesktop.DBus.Error.AccessDenied
Jun 12 11:01:32 rock-5b pipewire[1965]: mod.rt: could not set nice-level to -11: Permission denied
Jun 12 11:01:32 rock-5b pipewire[1965]: mod.rt: RTKit error: org.freedesktop.DBus.Error.AccessDenied
Jun 12 11:01:32 rock-5b pipewire[1965]: mod.rt: could not make thread 1989 realtime using RTKit: Permission denied




## 

pw-cli ls Module | grep pipe
 		module.name = "libpipewire-module-rt"
 		module.name = "libpipewire-module-protocol-native"
 		module.name = "libpipewire-module-profiler"
 		module.name = "libpipewire-module-metadata"
 		module.name = "libpipewire-module-spa-device-factory"
 		module.name = "libpipewire-module-spa-node-factory"
 		module.name = "libpipewire-module-client-node"
 		module.name = "libpipewire-module-client-device"
 		module.name = "libpipewire-module-portal"
 		module.name = "libpipewire-module-access"
 		module.name = "libpipewire-module-adapter"
 		module.name = "libpipewire-module-link-factory"
 		module.name = "libpipewire-module-session-manager"
