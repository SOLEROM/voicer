# pipe 2 example : module-filter-chain



    grabs audio straight from your USB mic,

    runs arbitrary DSP inside the PipeWire graph (no FIFOs, no extra user processes), and

    records the post-processed stream to one continuous WAV file.


1 What is module-filter-chain?

    It is a built-in PipeWire module that spawns two streams:

        a capture stream that receives audio from some node (your mic, another app, …);

        a playback stream that sends the filtered result onward.

    Between those streams you define an arbitrary graph of filters: built-in primitives (copy, limiter, mix, …), any LADSPA/LV2 plug-in on the system, or both.
    Ubuntu Manpages

    The resulting node can appear as:

        a virtual source (apps see it as a microphone),

        a virtual sink (apps see it as speakers/headphones), or

        a passive “insert” node that you wire in between two real devices.
        sanchayanmaity.pages.freedesktop.org

Because everything is still inside PipeWire, you avoid context-switches, buffer copies, or temporary files.


##

sudo apt install pipewire-audio
sudo apt install libspa-0.2-modules
sudo apt install swh-plugins




//is the module present?
sudo find / -name 'libpipewire-module-filter-chain.so' 2>/dev/null
/usr/lib/aarch64-linux-gnu/pipewire-0.3/libpipewire-module-filter-chain.so

//get the node name
pw-cli ls Node | grep alsa_input


cp 20-mic-analyzer.conf  /home/rock/.config/pipewire/pipewire.conf.d/20-mic-analyzer.conf

//debug
systemctl --user set-environment PIPEWIRE_DEBUG=3
journalctl --user -u pipewire -f &






// reload
systemctl --user restart pipewire wireplumber
pw-top                              # the new node appears green when active
pw-cli ls Node | grep mic-analyzer  # shows both capture & playback ends



/etc/security/limits.d/90-realtime.conf
@audio    -  rtprio   95
@audio    -  nice     -11
@audio    -  memlock  unlimited
@realtime -  rtprio   95
@realtime -  memlock  unlimited



//test
pw-cat --record --target mic-analyzer /tmp/a

