# Project SELMA - SELbstfahrendes Modell Auto

![SELMA](advertising/SELMA_Testfahrt.gif)

Das Projekt entstand im Zusammenhang des Integrationsseminares im Studium Data Science an der Dualen Hochschule Baden-Württemberg (DHBW) Mannheim. Eine erweiterte Dokumentation ist als wissenschaftliche Ausarbeitung als PDF hinterlegt.

## Getting Started

### Step 1 - Install all dependencys

Therefore simply run the <span>install.sh</span>
This will install all needed libaries, that can be found in <span>dependencies.txt</span> and <span>apt.txt</span>

### Step 2 - Install the pi to the car

Therefor you need to connect the camera to the pi and change the wiring harness of the car as shown below:

<img src="doku/Schaltplan.jpg" alt="drawing" width="400"/>


### Step 3 - Run the car

Now you can use the <span>run_car.py</span> or main<span>.py</span> script to either just contoll the car or also see the detection in the local netwerk via localhost:5000/.


## Built With

* [OpenCV](https://opencv.org)
* [YOLO](https://pjreddie.com/darknet/yolo/)
* [RPi.GPIO](www.raspberrypi.org/)
* [Flask](https://palletsprojects.com/p/flask/)


## Authors

* **Jan Brebeck** - *PiCar* - [Brebeck-Jan](https://github.com/Brebeck-Jan)
* **Andreas Bernrieder** - *Object Detection* - [Phantomias3782](https://github.com/Phantomias3782)
* **Simon Scapan** - *Lane Detection* - [SimonScapan](https://github.com/SimonScapan)
* **Thorsten Hilbradt** - *Contribution* - [Thorsten-H](https://github.com/Thorsten-H)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## Acknowledgments

Special Thanks to:
*  Soumya Ranjan Behera and his [Lane Line Detection](https://www.kaggle.com/soumya044/lane-line-detection/notebook) which was used as a Blueprint for this Lane Detection Pipeline.
* EbenKouao for his [Pi Camera Stream Flask](https://github.com/EbenKouao/pi-camera-stream-flask) which was used to implement our livestream.
