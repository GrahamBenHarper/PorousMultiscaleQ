// Multiscale Q1 implementation in deal.II
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace dealii;

/* Constructor
 * -Set polynomial degree
 * -Construct finite element space
 * -Construct DOF handler
 */

/* Setup
 * -Generate mesh
 * -Distribute DOFs
 * -Initialize global matrix
 * -Initialize solution vector
 * -Initialize rhs vector
 */

// Note: the assembly is hard to split up due to requirements for access to the 
/* Assemble system
 * -No need for global gaussian quadrature
 * -Initialize DOFs per cell, etc.
 * -Initialize local matrix and rhs
 * -Loop over elements
 *  -Collect the four element vertices
 *  -Generate a subgrid using those vertices, perform ___ refinements
 *  -Generate a finite element space on that triangulation
 *  -Generate a DOF handler
 *  -Loop over subgrid elements
 *  -Compute the subgrid's grad-grad matrix (will not change based on snapshot)
 *  -Solve each snapshot (set constraints based on basis functions from global scale) (use CG?)
 *  -Compute the multiscale basis function inner products and assemble them into the global problem
 *  -Compute the multiscale basis function inner product against f
 */



/* Solve
 * -Should be solvable with CG
 * -
 */

/* Compute errors
 * -Loop over elements
 * -Recompute the mesh
 * -Use the 
 */

/* Output results
 * -???
 * -Can output individual snapshots for now
 */



// Define the RHS for various choices
// Here we will use sin(pi*x)*sin(pi*y) as the exact solution, meaning the RHS is 2*pi*pi*sin(pi*x)*sin(pi*y)
template <int dim>
class RightHandSide_SinSin : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
      const unsigned int component = 0) const override;
};

template <int dim>
double RightHandSide_SinSin<dim>::value(const Point<dim> &p,
    const unsigned int /*component*/) const
    {
  double pi = numbers::PI;

  double return_value = 1;
  for(unsigned int d=0; d<dim; ++d)
    return_value *= std::sin(pi * p[d]);

  return_value *= dim*pi*pi;

  return return_value;
    }



// Define the exact solution for various choices
// Here we will use sin(pi*x)*sin(pi*y) as the exact solution.
template <int dim>
class ExactSolution_SinSin : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
      const unsigned int component = 0) const override;
};

template <int dim>
double ExactSolution_SinSin<dim>::value(const Point<dim> &p,
    const unsigned int /*component*/) const
    {
  double pi = numbers::PI;

  double return_value = 1;
  for(unsigned int d=0; d<dim; ++d)
    return_value *= std::sin(pi * p[d]);

  return return_value;
    }



// Define the boundary values for the multiscale Q1 problem
// There has to be an easier way to do this since the deal.II library uses the pullback for all finite element computations
// Note: This is so bad!!!
template <int dim>
class BoundaryValues_MS : public Function<dim>
{
public:
  BoundaryValues_MS(std::vector<Point<dim>> subgrid_corners,
      const unsigned int shape_index)
: vertices(subgrid_corners)
, index(shape_index)
{
    FullMatrix<double> coeffs(4,4); // bilinear_coefficients
    Vector<double> rhs_a(4);
    Vector<double> rhs_b(4);

    // basically, I'm constructing the bilinear pullback map for the coarse cell so that I can use this to impose boundary conditions
    // xhat = a_0 + a_1*x + a_2*y + a_3*x*y;
    // yhat = b_0 + b_1*x + b_2*y + b_3*x*y;
    // compute a, b on construction and save them
    coeffs(0,0) = 1; coeffs(0,1) = subgrid_corners[0][0]; coeffs(0,2) = subgrid_corners[0][1]; coeffs(0,3) = subgrid_corners[0][0]*subgrid_corners[0][1];
    coeffs(1,0) = 1; coeffs(1,1) = subgrid_corners[1][0]; coeffs(1,2) = subgrid_corners[1][1]; coeffs(1,3) = subgrid_corners[1][0]*subgrid_corners[1][1];
    coeffs(2,0) = 1; coeffs(2,1) = subgrid_corners[2][0]; coeffs(2,2) = subgrid_corners[2][1]; coeffs(2,3) = subgrid_corners[2][0]*subgrid_corners[2][1];
    coeffs(3,0) = 1; coeffs(3,1) = subgrid_corners[3][0]; coeffs(3,2) = subgrid_corners[3][1]; coeffs(3,3) = subgrid_corners[3][0]*subgrid_corners[3][1];

    rhs_a(1) = 1; rhs_a(3) = 1; // RHS for x component is (0,1,0,1)
    rhs_b(2) = 1; rhs_b(3) = 1; // RHS for y component is (0,0,1,1)

    coeffs.gauss_jordan(); // coeffs = inverse(coeffs)

    // initialize the vectors
    a.reinit(4);
    b.reinit(4);

    // compute the map coefficients
    coeffs.vmult(a,rhs_a); // a = coeffs*rhs_a
    coeffs.vmult(b,rhs_b); // b = coeffs*rhs_b
}

  virtual double value(const Point<dim> & p,
      const unsigned int component = 0) const override;

private:
  const std::vector<Point<dim>> vertices;
  const unsigned int index;
  Vector<double> a; // pullback_coefficients_x;
  Vector<double> b; // pullback_coefficients_y;
};

template <int dim>
double BoundaryValues_MS<dim>::value(const Point<dim> &p,
    const unsigned int /*component*/) const
    {
  const double x = p[0];
  const double y = p[1];
  const double xhat = a[0] + a[1]*x + a[2]*y + a[3]*x*y;
  const double yhat = b[0] + b[1]*x + b[2]*y + b[3]*x*y;

  Vector<double> corner_values;
  corner_values.reinit(4); // there are 4 corners. all are 0 except corner number "index"
  corner_values(index) = 1;

  const double shape_value = corner_values(0)*(1-xhat)*(1-yhat)
                               + corner_values(1)*xhat*(1-yhat)
                               + corner_values(2)*(1-xhat)*yhat
                               + corner_values(3)*xhat*yhat;

  return shape_value;
    }



// Place everything in a class
// Assuming dim=spacedim for this
template <int dim>
class PorousMultiscale
{
public:
  // constructor setup
  PorousMultiscale(const unsigned int polynomial_degree, 
      const unsigned int coarse_refinements,
      const unsigned int fine_refinements);
  // run the private steps
  void run();

private:
  // the steps for execution in order
  void setup();
  void assemble();
  void solve();
  void compute_errors();
  void output_results();

  // the member variables on the coarse scale
  Triangulation<dim>        coarse_mesh;
  FE_Q<dim>                 coarse_fe;
  DoFHandler<dim>           coarse_dof_handler;
  SparsityPattern           coarse_sparsity_pattern;
  SparseMatrix<double>      coarse_matrix;
  AffineConstraints<double> coarse_constraints;
  Vector<double>            coarse_solution;
  Vector<double>            coarse_rhs;
  std::vector<Vector<double>> snapshot_solutions;

  // store the information passed to the constructor
  int poly_deg;
  int num_coarse_refinements;
  int num_fine_refinements;
};



// construct the class and also output the information that was passed to it for the log
template <int dim>
PorousMultiscale<dim>::PorousMultiscale(const unsigned int polynomial_degree, 
    const unsigned int coarse_refinements,
    const unsigned int fine_refinements)
    : coarse_fe(polynomial_degree)
      , coarse_dof_handler(coarse_mesh)
      , poly_deg(polynomial_degree)
      , num_coarse_refinements(coarse_refinements)
      , num_fine_refinements(fine_refinements)
      {
  std::cout << "============================" 
      << std::endl
      << "PorousMultiscale with"
      << " polynomial degree = " << polynomial_degree
      << ", coarse refinements = " << coarse_refinements
      << ", and fine refinements = " << fine_refinements
      << std::endl
      << "============================"
      << std::endl;
      }



// setup all of the information for the problem
template <int dim>
void PorousMultiscale<dim>::setup()
{
  // setup mesh
  GridGenerator::hyper_cube(coarse_mesh);
  coarse_mesh.refine_global(num_coarse_refinements);

  // setup dof handler
  coarse_dof_handler.distribute_dofs(coarse_fe);

  // setup vectors
  coarse_solution.reinit(coarse_dof_handler.n_dofs());
  coarse_rhs.reinit(coarse_dof_handler.n_dofs());

  // setup sparsity pattern and global matrix
  DynamicSparsityPattern dsp(coarse_dof_handler.n_dofs(), coarse_dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(coarse_dof_handler, dsp);
  coarse_sparsity_pattern.copy_from(dsp);
  coarse_matrix.reinit(coarse_sparsity_pattern);

  std::cout << "Number of active cells: " << coarse_mesh.n_active_cells()
                << std::endl
                << "Total number of cells: " << coarse_mesh.n_cells()
                << std::endl
                << "Number of multiscale basis functions: " << coarse_dof_handler.n_dofs()
                << std::endl;
}



// assemble the global and local problems; careful for memory leaks!
template <int dim>
void PorousMultiscale<dim>::assemble()
{
  RightHandSide_SinSin<dim> right_hand_side;
  // global mesh work
  // do nothing for now

  const unsigned int snapshots_per_cell = 4; // change this later to allow more extensibility
  bool drew_output = false;

  // loop over the coarse cells
  for (const auto &coarse_cell : coarse_dof_handler.active_cell_iterators())
  {
    // grab corners of the coarse cell
    std::vector<Point<dim>> subgrid_corners;
    for (unsigned int i=0; i<= GeometryInfo<dim>::vertices_per_cell; ++i)
      subgrid_corners.push_back(coarse_cell->vertex(i));

    ////////////////////////////////////////////////////////////////////////////
    // Start the subgrid problem
    ////////////////////////////////////////////////////////////////////////////

    // construct the fine grid using those corners
    Triangulation<dim> fine_mesh;
    GridGenerator::general_cell(fine_mesh, subgrid_corners);
    fine_mesh.refine_global(num_fine_refinements);

    // generate a fine dof handler, sparsity pattern, vectors, and matrix
    // generate gaussian quadrature, fe_values, and more
    FE_Q<dim>                 fine_fe(poly_deg); // probably not necessary
    DoFHandler<dim>           fine_dof_handler(fine_mesh);
    SparsityPattern           fine_sparsity_pattern;
    SparseMatrix<double>      fine_matrix;
    AffineConstraints<double> fine_constraints;
    Vector<double>            fine_solution;
    Vector<double>            fine_rhs;

    // setup dof handler
    fine_dof_handler.distribute_dofs(fine_fe);

    // setup vectors
    fine_rhs.reinit(fine_dof_handler.n_dofs());

    //setup matrix
    DynamicSparsityPattern dsp(fine_dof_handler.n_dofs(), fine_dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(fine_dof_handler, dsp);

    // setup sparsity pattern
    fine_sparsity_pattern.copy_from(dsp);
    fine_matrix.reinit(fine_sparsity_pattern);

#if false
    std::cout << "  Subgrid number of active cells: " << fine_mesh.n_active_cells()
                    << std::endl
                    << "  Subgrid total number of cells: " << fine_mesh.n_cells()
                    << std::endl
                    << "  Subgrid number of DOFs: " << fine_dof_handler.n_dofs()
                    << std::endl;
#endif

    // setup degree 2 quadrature and access to shape functions using FEValues
    QGauss<dim> quadrature_formula(2);
    FEValues<dim> fe_values(fine_fe, quadrature_formula,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    // count variables for loops
    const unsigned int dofs_per_cell   = fine_fe.dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula.size();

    // setup elementwise contributions
    FullMatrix<double> fine_local_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // construct the matrix for this subgrid once, then solve depending on different boundary conditions
    for (const auto &fine_cell : fine_dof_handler.active_cell_iterators())
    {
      fe_values.reinit(fine_cell);
      fine_local_matrix = 0;

      for (unsigned int q = 0; q < n_q_points; ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q);

            fine_local_matrix(i, j) += (grad_phi_i * grad_phi_j) * fe_values.JxW(q);
          }
        }

      fine_cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          fine_matrix.add(local_dof_indices[i],
              local_dof_indices[j],
              fine_local_matrix(i, j));

      for (unsigned int i = 0; i < 4; ++i)
        for (unsigned int j = 0; j < 4; ++j)
          std::cout << "Fine local matrix(" << i << "," << j << ") = " << fine_local_matrix(i,j) << std::endl;
    }

    // loop over snapshots, since all that changes is the boundary conditions
    for(unsigned int i_snapshot = 0; i_snapshot < snapshots_per_cell; ++i_snapshot)
    {
      // copy the mostly complete matrix, but apply BCs
      SparseMatrix<double> fine_snapshot_matrix;
      fine_snapshot_matrix.reinit(fine_sparsity_pattern);
      fine_snapshot_matrix.copy_from(fine_matrix);

      // setup fine solution
      fine_solution.reinit(fine_dof_handler.n_dofs());

      // setup fine RHS
      fine_rhs.reinit(fine_dof_handler.n_dofs());

      // apply boundary values
      std::map<types::global_dof_index, double> fine_boundary_values;
      VectorTools::interpolate_boundary_values(fine_dof_handler,
          0,
          BoundaryValues_MS<dim>(subgrid_corners, i_snapshot),
          fine_boundary_values);
      MatrixTools::apply_boundary_values(fine_boundary_values,
          fine_snapshot_matrix,
          fine_solution,
          fine_rhs);

      // solve this snapshot
      SolverControl            solver_control(1000, 1e-12);
      SolverCG<Vector<double>> cg(solver_control);
      PreconditionSSOR<SparseMatrix<double>> preconditioner;
      preconditioner.initialize(fine_snapshot_matrix, 1.2);
      cg.solve(fine_snapshot_matrix, fine_solution, fine_rhs, preconditioner);
      snapshot_solutions.push_back(fine_solution); // save the snapshot solution. for now we are saving them in order in snapshot_solutions.


      // Don't keep this forever!! We're just testing results here
      if(false)
      {
        GridOut           grid_out;
        GridOutFlags::Svg svg_flags;

        svg_flags.coloring           = GridOutFlags::Svg::level_number;
        svg_flags.label_boundary_id  = true;
        svg_flags.background         = GridOutFlags::Svg::transparent;
        svg_flags.label_level_number = true;
        svg_flags.label_cell_index   = true;
        svg_flags.draw_legend        = true;
        svg_flags.draw_colorbar      = true;

        grid_out.set_flags(svg_flags);
        std::ofstream grid_name("fine_mesh.svg");
        grid_out.write_svg(fine_mesh, grid_name);

        DataOut<dim> data_out;
        data_out.attach_dof_handler(fine_dof_handler);
        data_out.add_data_vector(fine_solution, "solution");
        data_out.build_patches();
        std::ofstream output("fine_solution_0.vtk");
        data_out.write_vtk(output);

        drew_output = true;
      }
    }


    // loop over this again to compute the global contributions of the multiscale basis functions before we toss local data
    FullMatrix<double> coarse_local_matrix(4, 4);
    Vector<double>     local_rhs(4);
    for (const auto &fine_cell : fine_dof_handler.active_cell_iterators())
    {
      coarse_local_matrix = 0;
      local_rhs = 0;

      // doing this the careful way... I think it can be done faster by just multiplying entries of fine_matrix with values from snapshot_solutions
      fe_values.reinit(fine_cell);
      fine_cell->get_dof_indices(local_dof_indices);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const Point<dim> x_q = fe_values.quadrature_point(q);
        double right_hand_side_value = right_hand_side.value(x_q);
        for (unsigned int i_snapshot = 0; i_snapshot < 4; ++i_snapshot)
          for (unsigned int j_snapshot = 0; j_snapshot < 4; ++j_snapshot)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const Tensor<1, dim> grad_phi_i = snapshot_solutions[snapshot_solutions.size() - 4 + i_snapshot][local_dof_indices[i]]*fe_values.shape_grad(i, q);
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                const Tensor<1, dim> grad_phi_j = snapshot_solutions[snapshot_solutions.size() - 4 + j_snapshot][local_dof_indices[j]]*fe_values.shape_grad(j, q);
                coarse_local_matrix(i_snapshot, j_snapshot) += (grad_phi_i * grad_phi_j) * fe_values.JxW(q);


                local_rhs(j) += (snapshot_solutions[snapshot_solutions.size() - 4 + j_snapshot][local_dof_indices[j]]*fe_values.shape_value(j, q) * right_hand_side_value) *
                    fe_values.JxW(q);
              }
            }
      }
    }

    coarse_cell->get_dof_indices(local_dof_indices);
    for (unsigned int i = 0; i < 4; ++i)
      for (unsigned int j = 0; j < 4; ++j)
        coarse_matrix.add(local_dof_indices[i],
                          local_dof_indices[j],
                          coarse_local_matrix(i, j));
    for (unsigned int j = 0; j < 4; ++j)
      coarse_rhs(local_dof_indices[j]) += local_rhs(j);


    for (unsigned int i = 0; i < 4; ++i)
      for (unsigned int j = 0; j < 4; ++j)
        std::cout << "Coarse local matrix(" << i << "," << j << ") = " << coarse_local_matrix(i,j) << std::endl;

    std::cout << "Coarse local rhs = " << local_rhs << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    // End the subgrid problem
    ////////////////////////////////////////////////////////////////////////////


    // assemble the multiscale basis functions into the coarse system

    std::map<types::global_dof_index, double> coarse_boundary_values;
    VectorTools::interpolate_boundary_values(coarse_dof_handler,
        0,
        ExactSolution_SinSin<dim>(),
        coarse_boundary_values);

    MatrixTools::apply_boundary_values(coarse_boundary_values,
        coarse_matrix,
        coarse_solution,
        coarse_rhs);
  }

  std::cout << "Snapshots saved: " << snapshot_solutions.size() << std::endl;
  for (unsigned int i = 0; i < snapshot_solutions.size(); ++i)
  {
    std::cout << "Snapshot solution " << i << " = " << snapshot_solutions[i] << std::endl;
  }
}



// solve the resulting linear system
template <int dim>
void PorousMultiscale<dim>::solve()
{
  // default to 1000 iterations, 1e-12 error settings
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> cg(solver_control);

  // setup a simple preconditioner for now
  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(coarse_matrix, 1.2);

  // solve
  cg.solve(coarse_matrix, coarse_solution, coarse_rhs, preconditioner);
}



// compute errors for the obtained solution
template <int dim>
void PorousMultiscale<dim>::compute_errors()
{
  Vector<double> cell_L2_error;
  cell_L2_error.reinit(coarse_mesh.n_active_cells());
  ExactSolution_SinSin<dim> exact_solution;

  // const unsigned int snapshots_per_cell = 4; // change this later to allow more extensibility

  unsigned int k=0; // count coarse cells

  // loop over the coarse cells again
  for (const auto &coarse_cell : coarse_dof_handler.active_cell_iterators())
  {
    // grab corners of the coarse cell again
    std::vector<Point<dim>> subgrid_corners;
    for (unsigned int i=0; i<= GeometryInfo<dim>::vertices_per_cell; ++i)
      subgrid_corners.push_back(coarse_cell->vertex(i));

    // construct the fine grid using those corners
    Triangulation<dim> fine_mesh;
    GridGenerator::general_cell(fine_mesh, subgrid_corners);
    fine_mesh.refine_global(num_fine_refinements);

    // generate a fine dof handler
    // generate gaussian quadrature, fe_values, and more
    FE_Q<dim>                 fine_fe(poly_deg); // probably not necessary
    DoFHandler<dim>           fine_dof_handler(fine_mesh);
    SparsityPattern           fine_sparsity_pattern;

    // setup dof handler
    fine_dof_handler.distribute_dofs(fine_fe);

    // setup degree 3 quadrature and access to shape functions using FEValues
    QGauss<dim> quadrature_formula(3);
    FEValues<dim> fe_values(fine_fe, quadrature_formula,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    // count variables for loops
    const unsigned int dofs_per_cell   = fine_fe.dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula.size();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // construct the matrix for this subgrid once, then solve depending on different boundary conditions
    for (const auto &fine_cell : fine_dof_handler.active_cell_iterators())
    {
      fe_values.reinit(fine_cell);
      coarse_cell->get_dof_indices(local_dof_indices);


      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double numerical_solution_value = 0;
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          // use all 4 snapshots for this shape function
          // then use all shape functions on this cell
          numerical_solution_value += (snapshot_solutions[4*k][local_dof_indices[i]]
                                      +snapshot_solutions[4*k+1][local_dof_indices[i]]
                                      +snapshot_solutions[4*k+2][local_dof_indices[i]]
                                      +snapshot_solutions[4*k+3][local_dof_indices[i]])*fe_values.shape_value(i, q);
        }

        const Point<dim> x_q = fe_values.quadrature_point(q);
        const double exact_solution_value = exact_solution.value(x_q);

        cell_L2_error(k) += (numerical_solution_value - exact_solution_value)*(numerical_solution_value - exact_solution_value) * fe_values.JxW(q);
      }
    }

    cell_L2_error(k) = std::sqrt(cell_L2_error(k));

    ++k;
  }
  double L2_error = 0;
  for (unsigned int k = 0; k < cell_L2_error.size(); ++k)
    L2_error += cell_L2_error(k)*cell_L2_error(k);
  L2_error = std::sqrt(L2_error);

  std::cout << "L2 error: " << L2_error << std::endl;
}



// output the global solution
template <int dim>
void PorousMultiscale<dim>::output_results()
{
  // TODO
  // DataOut likes to have only 1 DOF handler and only 1 solution
  // I need to brainstorm an appropriate way to take many fine DOF handlers and combine their numerical outputs into a single large output
}



// run everything
template <int dim>
void PorousMultiscale<dim>::run()
{
  setup();
  assemble();
  solve();
  compute_errors();
  output_results();
}



// the main routine
int main()
{
  try
  {
    std::cout.precision(6);
    // run the 2d problem with Q1, 4 coarse and 4 fine refinements
    PorousMultiscale<2>(1,2,2).run();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
        << std::endl
        << "----------------------------------------------------"
        << std::endl;
    std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
        << std::endl
        << "----------------------------------------------------"
        << std::endl;
    std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
    return 1;
  };
  return 0;
}
