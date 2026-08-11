#pragma once

#include <utility>
#include <vector>

namespace evaluador {

// Defuzzificacion por promedio ponderado (equivalente al metodo de
// altura / height defuzzification para consecuentes singleton Sugeno):
//   score = sum(fuerza_i * valor_i) / sum(fuerza_i)
//
// Con el diseño de MotorReglas + Fuzzificador de este proyecto, la
// suma de fuerzas siempre da 1.0 (las 3 particiones difusas de la
// etapa 5 estan construidas para sumar exactamente 1 en todo el rango),
// asi que en la practica esto es un promedio ponderado directo — pero
// se divide explicitamente por la suma de fuerzas de todas formas, por
// si algun dia MotorReglas se usa con una tabla de reglas donde eso ya
// no sea cierto.
double defuzzificar(const std::vector<std::pair<double, double>>& disparos);

} // namespace evaluador
