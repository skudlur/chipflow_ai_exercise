#!/bin/sh
if cat /etc/os-release | grep -w -E "ID=fedora";
then
    echo "Installing dependencies for Fedora...";
    echo "Installing GitHub CLI (gh) utility...";
    sudo dnf install gh;
    echo "Installed gh utility successfully!";
else
    if cat /etc/os-release | grep -w -E "ID=ubuntu";
    then
	echo "Installing dependencies for Ubuntu...";
	echo "Installing GitHub CLI (gh) utility...";
	sudo apt install gh;
	echo "Installed gh utility successfully!";
    fi
fi
