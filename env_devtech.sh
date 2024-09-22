#!/bin/bash
export MELLANOX_VISIBLE_DEVICES=void
enroot create --name dt /lustre/fsw/coreai_devtech_all/hwaugh/containers/devtech.sqsh
enroot start --mount /lustre/fsw/coreai_devtech_all/hwaugh/:/lustre/fsw/coreai_devtech_all/hwaugh/ dt
#enroot start --mount /lustre/fsw/coreai_devtech_all/hwaugh/:/lustre/fsw/coreai_devtech_all/hwaugh/ --mount /usr/sbin/:/usr/sbin/ dt
