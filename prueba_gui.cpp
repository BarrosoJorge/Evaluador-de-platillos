// prueba_gui.cpp — diagnostico minimo, no forma parte del pipeline.
//
// Compilar:
//   g++ -std=c++17 prueba_gui.cpp -o prueba_gui `pkg-config --cflags --libs opencv4`
//
// Correr:
//   ./prueba_gui
//
// Que esperar si el GUI de OpenCV funciona bien:
//   Se abre una ventana con un rectangulo de color. Debes poder
//   arrastrar el mouse para dibujar un ROI, y la terminal debe
//   imprimir las coordenadas que elegiste DESPUES de que presiones
//   ESPACIO/ENTER — no antes.
//
// Si en vez de eso la terminal imprime el resultado casi
// inmediatamente sin que hayas alcanzado a arrastrar nada, el
// problema es el backend grafico (X server / WSLg), no el codigo del
// pipeline.

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat imagen(400, 600, CV_8UC3, cv::Scalar(40, 40, 40));
    cv::rectangle(imagen, cv::Rect(150, 100, 300, 200), cv::Scalar(0, 180, 255), -1);
    cv::putText(imagen, "Arrastra un ROI y presiona ESPACIO", cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    std::cout << "Abriendo ventana de prueba..." << std::endl;
    cv::Rect roi = cv::selectROI("Prueba GUI", imagen, false, false);
    cv::destroyAllWindows();

    std::cout << "ROI seleccionado: x=" << roi.x << " y=" << roi.y
              << " w=" << roi.width << " h=" << roi.height << std::endl;

    if (roi.empty()) {
        std::cout << "\n>>> ROI vacio. Si esto aparecio casi instantaneo, "
                     "sin darte tiempo de arrastrar nada, el GUI no esta "
                     "funcionando correctamente en este entorno." << std::endl;
    } else {
        std::cout << "\n>>> Funciono correctamente." << std::endl;
    }

    return 0;
}