#pragma once

#include "CalculadorPercentiles.hpp"

namespace evaluador {

// Grados de pertenencia a los 3 terminos linguisticos definidos para
// esta etapa. No estan forzados a sumar 1.0 (asi es como funcionan
// normalmente las particiones difusas triangulares/trapezoidales con
// traslape), pero con este diseño especifico (ver fuzzificar()) la
// suma siempre da 1.0 excepto en los extremos, donde satura.
struct GradosPertenencia {
    double similar = 0.0;
    double algoDistinto = 0.0;
    double muyDistinto = 0.0;
};

// Fuzzifica una distancia de Mahalanobis usando las anclas (p33, p50,
// p66) calculadas para esa dimension en esa sesion especifica — el
// mismo criterio que se discutio: "similar" = por debajo del percentil
// 33, "muy distinto" = por encima del percentil 66, con transicion
// triangular en medio (sin necesidad de labels humanos).
//
// Diseño de las 3 funciones (todas en terminos de d = distancia):
//   similar(d):      1 si d<=p33; rampa descendente hasta 0 en p50
//   algo_distinto(d): 0 en d<=p33 o d>=p66; triangular con pico en p50
//   muy_distinto(d):  0 si d<=p50; rampa ascendente hasta 1 en p66
//
// Caso degenerado: si p33==p50==p66 (ej. sesion con muy pocos alumnos,
// o distancias identicas), no hay forma de definir rampas — se cae a
// una regla simple: distancia menor o igual al ancla = similar=1,
// mayor = muy_distinto=1. Se registra un WARN para que quede claro que
// la fuzzificacion no fue "real" en ese caso.
GradosPertenencia fuzzificar(double distancia, const AnclasPercentiles& anclas);

} // namespace evaluador
