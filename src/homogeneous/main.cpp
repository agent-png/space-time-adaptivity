#include "H_Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Heat::dim;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };
  double a; //=?????
  int N; //=?????
  Point<dim> x0; //=????
  double sigma; //=?????
  const auto g  = [&]( const double  &t) {
    return (std::exp(-a * std::cos(2*N*M_PI*t)))/(std::exp(a));
  };
  const auto h = [&]( const Point<dim> &p) {
    return std::exp(-((p-x0)*(p-x0)/std::pow(sigma,2)));
  };

  const auto f  = [&](const Point<dim>  &p, const double  &t) {
    return g(t)*h(p);
  };

  Heat problem(/*mesh_filename = */ "../mesh/mesh-cube-10.msh",
               /* degree = */ 1,
               /* T = */ 1.0,
               /* theta = */ 0.0,
               /* delta_t = */ 0.0025,
               mu,
               f);

  problem.run();

  return 0;
}