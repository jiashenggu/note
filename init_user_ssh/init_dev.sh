# initialize a new development machine
apt-get update
apt-get install htop
apt-get install nload
bash _add_user.sh gujiasheng 0 "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCYsPpUd4fCqk+4HzggFMU/RidZn1u9q/Va2ISw/M7RkzwxuZ4UQD/T99lwKSDMqB1ByWL2domIxk1G/2/sXWKT3UELzamO3y2m3vV95dMbEDBoAFzb77z2QQ/Bhjus58oGCC2NByc0GKxNwkoG4JOGscvQ5hDYRzhyOekX/72bYl0Ylfcg8n5aYhyEDcZt2IA2PHqve1u0f3/ak6Q+NzSTb+IVWVLECL8pKGqzK2pAG5OHMNge/9h8d2u5Q1i0W+9W4T7cfNB+K/LaChlukNM5jTBRwtttnUmQzJN19StoIKSIKHVd7QHDghAvq9zdY4JoMsVxzmpNnXsatXa1q3oMK98NxKDeGuVAEzem16v0xK0JjhzHlDvRBzscP7ARW+v1EL0nqd5eAmL0LDnOztMIB1BlxuF8M9+/j0ylSSCiuB2FdRnomleZNtG7CxAtasVzsJgjeqyR7xOiJX7uuiQY9qZ+EERp+lxn1yEUeVc3IsO09ssqXvfFStq4n2A7D3c= scruple@DESKTOP-G9MF5S2" /ML-A100/team/mm g

# cd /home
# mkdir wangtao
# cd /home/wangtao
cd /ML-A100/team/mm/gujiasheng
# ln -s /ML-A100/team/mm/gujiasheng base
# ln -s base/anaconda3 anaconda3

# view /etc/passwd to check if the user is added