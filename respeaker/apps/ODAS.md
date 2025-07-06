# ODAS - Open embeddeD Audition System

ODAS stands for Open embeddeD Audition System. This is a library dedicated to perform sound source localization, tracking, separation and post-filtering. ODAS is coded entirely in C, for more portability, and is optimized to run easily on low-cost embedded hardware


* install:

```
sudo apt-get install libfftw3-dev libconfig-dev libasound2-dev libgconf-2-4
sudo apt-get install cmake
sudo apt-get install libpulse-dev

git clone https://github.com/introlab/odas.git
mkdir odas/build
cd odas/build
cmake ..
make
```

* ODAS Studio

```

sudo apt install nodejs
sudo apt install npm
git clone https://github.com/introlab/odas_web
cd odas_web

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
export NVM_DIR="$HOME/.nvm"
source "$NVM_DIR/nvm.sh"

nvm install 14
nvm use 14
npm install 


npm start

```