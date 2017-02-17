#include <QApplication>
#include <QtConcurrent>
#include <QFutureWatcher>
#include <QFileDialog>
#include <QShortcut>

#include <iostream>
#include <thread>


#ifndef SKEPU_PRECOMPILED
#define SKEPU_PRECOMPILED
#endif
#ifndef SKEPU_OPENCL
#define SKEPU_OPENCL
#endif
#include <skepu2.hpp>

#include "lodepng.h"
#include "skepuimg.h"
#include "ui_mainwindow.h"

namespace imgp = SkePUImageProcessing;

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(QMainWindow *parent = 0): QMainWindow(parent)
	{
		ui.setupUi(this);
		this->ui.cpuThreadsBox->setValue(imgp::getCPUThreads());
		this->ui.backendBox->setCurrentIndex(this->ui.backendBox->findText(QString::fromStdString(imgp::getBackend())));
		
		this->sk_stencil = skepu2::Matrix<float>(3, 3);
		
		this->filterWatcher = new QFutureWatcher<float>();
		connect(this->filterWatcher, SIGNAL(finished()), this, SLOT(filteringFinished()));
		
		this->loadWatcher = new QFutureWatcher<float>();
		connect(this->loadWatcher, SIGNAL(finished()), this, SLOT(loadingFinished()));
		
		connect(this->ui.hueSlider, SIGNAL(sliderMoved(int)), this, SLOT(on_hueButton_clicked(int)));
		
		QObject::connect(new QShortcut(QKeySequence("Ctrl+L"), this), SIGNAL(activated()), this, SLOT(tabToLoad()));
		QObject::connect(new QShortcut(QKeySequence("Ctrl+G"), this), SIGNAL(activated()), this, SLOT(tabToGenerate()));
		QObject::connect(new QShortcut(QKeySequence("Ctrl+F"), this), SIGNAL(activated()), this, SLOT(tabToBuiltin()));
		QObject::connect(new QShortcut(QKeySequence("Ctrl+U"), this), SIGNAL(activated()), this, SLOT(tabToCustom()));
		QObject::connect(new QShortcut(QKeySequence("Ctrl+B"), this), SIGNAL(activated()), this, SLOT(tabToBackend()));
	}
	
private slots:
	void on_gaussianButton_clicked()
	{
		if (!this->filtering)
		{
			this->showMessage("Blur filtering ...");
			this->filtering = true;
			int sigma = this->ui.blurRadiusBox->value();
			QFuture<float> future = QtConcurrent::run(imgp::gaussian<RGBPixel>, &this->sk_image, sigma);
			this->filterWatcher->setFuture(future);
		}
	}
	
	void tabToLoad()     { this->ui.tabWidget->setCurrentIndex(0); }
	void tabToGenerate() { this->ui.tabWidget->setCurrentIndex(1); }
	void tabToBuiltin()  { this->ui.tabWidget->setCurrentIndex(2); }
	void tabToCustom()   { this->ui.tabWidget->setCurrentIndex(3); }
	void tabToBackend()  { this->ui.tabWidget->setCurrentIndex(4); }
	
	void on_medianButton_clicked()
	{
		if (!this->filtering)
		{
			this->showMessage("Blur filtering ...");
			this->filtering = true;
			int radius = this->ui.blurRadiusBox->value();
			QFuture<float> future = QtConcurrent::run(imgp::median, &this->sk_image, radius);
			this->filterWatcher->setFuture(future);
		}
	}
	
	void on_hueButton_clicked(int)
	{
		if (!this->filtering)
		{
			this->showMessage("Hue adjustment ...");
			this->filtering = true;
			float hue = this->ui.hueSlider->value() * 3.1416f / 180.0f;
			QFuture<float> future = QtConcurrent::run(imgp::hue, &this->sk_image, hue);
			this->filterWatcher->setFuture(future);
		}
	}
	
	void on_mandelbrotButton_clicked()
	{
		if (!this->filtering)
		{
			this->showMessage("Mandelbrot generator ...");
			this->filtering = true;
			float scale = this->ui.scaleSlider->value() / 10.0;
			this->sk_image.resize(this->ui.heightBox->value(), this->ui.widthBox->value());
			QFuture<float> future = QtConcurrent::run(imgp::mandelbrot, &this->sk_image, scale);
			this->filterWatcher->setFuture(future);
		}
	}
	
	void on_edgeButton_clicked()
	{
		if (!this->filtering)
		{
			this->showMessage("Edge detection filtering ...");
			this->filtering = true;
			QFuture<float> future = QtConcurrent::run(imgp::edge_rgb, &this->sk_image);
			this->filterWatcher->setFuture(future);
		}
	}
	
	void on_grayscaleButton_clicked()
	{
		if (!this->filtering)
		{
			this->showMessage("Desaturating ...");
			this->filtering = true;
			float sat = this->ui.hueSlider->value() / 360.f;
			QFuture<float> future = QtConcurrent::run(imgp::desaturate, &this->sk_image, sat);
			this->filterWatcher->setFuture(future);
		}
	}
	
	void on_bwButton_clicked()
	{
		if (!this->filtering)
		{
			this->showMessage("Black/White filtering ...");
			this->filtering = true;
			QFuture<float> future = QtConcurrent::run(imgp::blackwhite, &this->sk_image);
			this->filterWatcher->setFuture(future);
		}
	}
	
	void on_stencilButton_clicked()
	{
		if (!this->filtering)
		{
			this->showMessage("Stencil filtering ...");
			this->filtering = true;
			float scaling = 1 / this->ui.coeffScaling->text().toFloat();
			this->readStencil();
			QFuture<float> future = QtConcurrent::run(imgp::stencil, &this->sk_image, &this->sk_stencil, scaling);
			this->filterWatcher->setFuture(future);
		}
	}
	
	void on_invertButton_clicked()
	{
		if (!this->filtering)
		{
			this->showMessage("Grayscale filtering ...");
			this->filtering = true;
			QFuture<float> future = QtConcurrent::run(imgp::invert<RGBPixel>, &this->sk_image);
			this->filterWatcher->setFuture(future);
		}
	}
	
	void on_browseButton_clicked()
	{
		QFileDialog dialog(this);
		dialog.setFileMode(QFileDialog::ExistingFile);
		dialog.setNameFilter(tr("Images (*.png *.xpm *.jpg)"));
		if (dialog.exec())
			this->ui.filenameField->setText(*dialog.selectedFiles().begin());
	}
	
	static float loadImage(std::string fileName, skepu2::Matrix<RGBPixel> *sk_img)
	{
		unsigned error;
		unsigned char *img;
		unsigned width = 0, height = 0;
		std::chrono::microseconds time = skepu2::benchmark::measureExecTime([&]
		{
			error = lodepng_decode_file(&img, &width, &height, fileName.c_str(), LCT_RGB, 8);
			if (!error)
			{
				*sk_img = std::move(skepu2::Matrix<RGBPixel>(height, width));
				memcpy(&(*sk_img)[0], &img[0], width * height * 3);
			}
			else { std::cerr << "ERROR!" << lodepng_error_text(error) << "\n"; }
		});
		return (!error) ? time.count() / 1E6 : -1;
	}
	
	void on_loadButton_clicked()
	{
		this->showMessage("Loading image ...");
		this->ui.loadButton->setEnabled(false);
		std::string filePath = this->ui.filenameField->text().toStdString();
		QFuture<float> future = QtConcurrent::run(loadImage, filePath, &this->sk_image);
		this->loadWatcher->setFuture(future);
	}
	
	void loadingFinished()
	{
		this->showMessage("Loading done.");
		float time = this->loadWatcher->result();
		std::cout << "Time: " << time << "\n";
		this->ui.loadButton->setEnabled(true);
		if (time >= 0)
		{
			this->displayImage();
		}
	}
	
	void filteringFinished()
	{
		this->showMessage("Filtering done.");
		std::cout << "Time: " << this->filterWatcher->result() << "\n";
		this->displayImage();
		this->filtering = false;
	}
	
	void on_backendBox_currentTextChanged(QString text)
	{
		this->showMessage("Changed backend!");
		imgp::setBackend(text.toStdString());
	}
	
	void on_cpuThreadsBox_valueChanged(int value)
	{
		this->showMessage("Changed number of OpenMP threads!");
		imgp::setCPUThreads(value);
	}
	
	void on_clearStencilButton_clicked()
	{
		this->ui.coeff0_0->setText("0.0");
		this->ui.coeff0_1->setText("0.0");
		this->ui.coeff0_2->setText("0.0");
		this->ui.coeff1_0->setText("0.0");
		this->ui.coeff1_1->setText("1.0");
		this->ui.coeff1_2->setText("0.0");
		this->ui.coeff2_0->setText("0.0");
		this->ui.coeff2_1->setText("0.0");
		this->ui.coeff2_2->setText("0.0");
	}

private:
	
	void readStencil()
	{
		this->sk_stencil(0, 0) = this->ui.coeff0_0->text().toFloat();
		this->sk_stencil(0, 1) = this->ui.coeff0_1->text().toFloat();
		this->sk_stencil(0, 2) = this->ui.coeff0_2->text().toFloat();
		this->sk_stencil(1, 0) = this->ui.coeff1_0->text().toFloat();
		this->sk_stencil(1, 1) = this->ui.coeff1_1->text().toFloat();
		this->sk_stencil(1, 2) = this->ui.coeff1_2->text().toFloat();
		this->sk_stencil(2, 0) = this->ui.coeff2_0->text().toFloat();
		this->sk_stencil(2, 1) = this->ui.coeff2_1->text().toFloat();
		this->sk_stencil(2, 2) = this->ui.coeff2_2->text().toFloat();
	}
	
	void displayImage()
	{
		this->sk_image.updateHost();
		QImage img(reinterpret_cast<unsigned char*>(&this->sk_image[0]),
			this->sk_image.total_cols(), this->sk_image.total_rows(), QImage::Format_RGB888);
		QPixmap pixmap;
		pixmap.convertFromImage(img);
		this->ui.imageView->setPixmap(pixmap);
	}
	
	void showMessage(std::string message)
	{
		ui.statusbar->showMessage(QString::fromStdString(message));
	}
	
	Ui::MainWindow ui;
	skepu2::Matrix<RGBPixel> sk_image;
	skepu2::Matrix<float> sk_stencil;
	QFutureWatcher<float> *filterWatcher, *loadWatcher;
	bool filtering = false;
};


#include "main.moc"

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	MainWindow main;
	main.show();
	return app.exec();
}
