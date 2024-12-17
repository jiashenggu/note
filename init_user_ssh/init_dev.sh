# initialize a new development machine
apt-get update
apt-get install htop
apt-get install nload
bash _add_user.sh gujiasheng 0 $ssh_key $HOME_PATH $USER_PASSWORD

cd $HOME_PATH
# ln -s $HOME_PATH base
# ln -s base/anaconda3 anaconda3

# view /etc/passwd to check if the user is added
