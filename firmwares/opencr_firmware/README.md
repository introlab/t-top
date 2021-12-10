# OpenCR Firmware

## Installation de l'IDE
1. Installer Arduino IDE ([ici](http://emanual.robotis.com/docs/en/parts/controller/opencr10/#install-on-linux));
2. Modifier le fichier `~/.arduino15/packages/OpenCR/hardware/OpenCR/1.4.9/libraries/DynamixelWorkbench/src/dynamixel_workbench_toolbox/dynamixel_driver.cpp`
    1. Mettre en commentaire le `delay` dans la méthode `bool writeRegister(uint8_t, uint16_t, uint16_t, uint8_t*, const char**);`    
    2. Mettre en commentaire le `delay` dans la méthode `bool writeRegister(uint8_t, const char*, int32_t, const char**);`    
    3. Mettre en commentaire le `delay` dans la méthode `bool writeOnlyRegister(uint8_t, uint16_t, uint16_t, uint8_t*, const char**);`    
    4. Mettre en commentaire le `delay` dans la méthode `bool writeOnlyRegister(uint8_t, const char*, int32_t, const char**);`
3. Copier-coller le fichier `100-opencr-custom.rules` dans `/etc/udev/rules.d/`<br />
`sudo cp ~/tabletop_ws/src/tabletop_robot/code/opencr_firmware/100-opencr-custom.rules /etc/udev/rules.d/100-opencr-custom.rules`

## Programmation de la carte
1. Ouvrir le projet dans Arduino IDE;
2. Sélectionner la carte OpenCR;
3. Téléverser le projet.
