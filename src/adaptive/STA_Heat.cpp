#include "STA_Heat.hpp"

void AdaptiveHeat::setup()
{
  pcout << "===============================================" << std::endl;

  // Create the initial mesh
  {
    pcout << "Initializing the mesh" << std::endl;

    GridGenerator::hyper_cube(mesh);
    //refining level
    mesh.refine_global(5);

    pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;

  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_Q<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGauss<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;
  
  setup_system();
}

void AdaptiveHeat::setup_system(){
  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
    pcout << "  N_levels = " << mesh.n_levels() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);


    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    pcout << "  Initializing the sparsity pattern" << std::endl;
    // deal.II tutorial 40 also uses SparsityTools::distribute_sparsity_pattern()
    // but TrilinosWrappers::SparsityPattern already creates a parallel sparsity pattern
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    // keep_constrained_dofs=false -> Do not treat constrained DoFs as independent unknowns.
    // Instead, the constraints are substituted into the matrix structure.
    DoFTools::make_sparsity_pattern(dof_handler,
                                    sparsity,
                                    constraints,
                                    /*keep_constrained_dofs=*/false); 
    sparsity.compress();

    pcout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity);

    pcout << "  Initializing vectors" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

void AdaptiveHeat::assemble()
{
  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  // Local matrix and vector.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs    = 0.0;

  // Evaluation of the old solution on quadrature nodes of current cell.
  std::vector<double> solution_old_values(n_q);

  // Evaluation of the gradient of the old solution on quadrature nodes of
  // current cell.
  std::vector<Tensor<1, dim>> solution_old_grads(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      // Evaluate the old solution and its gradient on quadrature nodes.
      fe_values.get_function_values(solution, solution_old_values);
      fe_values.get_function_gradients(solution, solution_old_grads);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double mu_loc = mu(fe_values.quadrature_point(q));

          const double f_old_loc =
            f(fe_values.quadrature_point(q), time - delta_t);
          const double f_new_loc = f(fe_values.quadrature_point(q), time);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Time derivative.
                  cell_matrix(i, j) += (1.0 / delta_t) *             //
                                       fe_values.shape_value(i, q) * //
                                       fe_values.shape_value(j, q) * //
                                       fe_values.JxW(q);

                  // Diffusion.
                  cell_matrix(i, j) +=
                    theta * mu_loc *                             //
                    scalar_product(fe_values.shape_grad(i, q),   //
                                   fe_values.shape_grad(j, q)) * //
                    fe_values.JxW(q);
                }

              // Time derivative.
              cell_rhs(i) += (1.0 / delta_t) *             //
                             fe_values.shape_value(i, q) * //
                             solution_old_values[q] *      //
                             fe_values.JxW(q);

              // Diffusion.
              cell_rhs(i) -= (1.0 - theta) * mu_loc *                   //
                             scalar_product(fe_values.shape_grad(i, q), //
                                            solution_old_grads[q]) *    //
                             fe_values.JxW(q);

              // Forcing term.
              cell_rhs(i) +=
                (theta * f_new_loc + (1.0 - theta) * f_old_loc) * //
                fe_values.shape_value(i, q) *                     //
                fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);

      constraints.distribute_local_to_global(cell_matrix,
                                       cell_rhs,
                                       dof_indices,
                                       system_matrix,
                                       system_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // Homogeneous Neumann boundary conditions: we do nothing.
}

void AdaptiveHeat::solve_time_step()
{
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(
    system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  //ReductionControl is a more flexible SolverControl extension
  ReductionControl solver_control(/* maxiter = */ 10000,
                                  /* tolerance = */ 1.0e-12,
                                  /* reduce = */ 1.0e-6);

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  solution_owned.reinit(dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);

  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
  pcout << solver_control.last_step() << " CG iterations" << std::endl;

  // 
  constraints.distribute(solution_owned);
  
}

bool AdaptiveHeat::refine_grid(const unsigned int min_grid_level,
                               const unsigned int max_grid_level)
{
  Vector<float> estimated_error_per_cell(mesh.n_active_cells());
  
  KellyErrorEstimator<dim>::estimate(
    dof_handler,
    QGauss<dim - 1>(fe->degree + 1),
    std::map<types::boundary_id, const Function<dim> *>(),
    solution,
    estimated_error_per_cell);
  
  double eta_norm = estimated_error_per_cell.l2_norm();
  pcout << "  Number of DoFs to refine = " << dof_handler.n_dofs()*0.25 << std::endl;
  pcout << "  Number of DoFs to coarsen = " << dof_handler.n_dofs()*0.03 << std::endl;
  pcout << "  Estimated error norm =  " <<std::scientific << eta_norm << std::endl;
  
  if(eta_norm < spatial_tol){
    /* do not refine the grid */
    pcout << "  Spatial error " << eta_norm 
          << " < tolerance " << spatial_tol 
          << " -> skip refinement.\n";
    return false; 
  } 

  /* changes to make refine and coarsen fraction dynamic */
  const double error_ratio = eta_norm / spatial_tol; 

  // how does it work
  // If error_ratio = 2, error is 2x the tol → use higher refine_fraction
  // if error_ratio = 1.1, error is a little higher than tol → use lower refine_fraction
  const double refine_fraction  = std::clamp(0.10 * error_ratio, 0.05, 0.30); //refine_fraction = 0.10 * error_ratio, 0.05 is the min and 0.30 is the max
  const double coarsen_fraction = std::clamp(0.01 / error_ratio, 0.005, 0.03);

  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
      mesh, 
      estimated_error_per_cell, 
      refine_fraction,
      coarsen_fraction,
      VectorTools::L1_norm);
  /**
   * EXPLANATION parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction fuction fields
   * 0.3 and 0.03 values explaination (I think it can be useful)
   * There is an estimated error for each cell:
   *  - error is high --> the cell needs more detail
   *  - error is low --> the cell is already fine
   * So:
   * - 0.3(refine fraction) --> the worst 30% of cells are split into smaller cells
   * - 0.03(coarsen fraction) --> the best 3% of cells are merged into bigger cells
   * 
   * This are default values.
   ** How to change them in the future(keep coarsen much smaller than refine (often 5-10x smaller)):
   *   - if error decreases too slowly, increase refine fraction (0.4 max-ish).
   *   - if mesh oscillates (refine/coarsen flip-flop), reduce coarsen fraction.
   *   - if DoFs explode, reduce refine fraction and/or add max refinement level.
   */

   

  // Enforce max and min/max refinement levels
  if (mesh.n_levels() > max_grid_level){
    for (const auto &cell : mesh.active_cell_iterators_on_level(max_grid_level)){
            cell->clear_refine_flag();
    }
  }
    
  for (const auto &cell : mesh.active_cell_iterators_on_level(min_grid_level)){
    cell->clear_coarsen_flag();
  }
   
  
  // Prepare solution transfer
  parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> solution_transfer(dof_handler);
  // previous_solution is the solution to transfer (tutorial-26)
  TrilinosWrappers::MPI::Vector previous_solution(solution);

  mesh.prepare_coarsening_and_refinement();
  solution_transfer.prepare_for_coarsening_and_refinement(previous_solution);

  // Execute refinement
  mesh.execute_coarsening_and_refinement();

  // Rebuild DoFs/matrix/vectors on new mesh
  setup_system();

  // Interpolate old solution onto the new DoF space and apply hanging-node and Dirichlet constraints for continuity.
  solution_transfer.interpolate(solution_owned);
  constraints.distribute(solution_owned);

  // Update ghosted vector
  solution = solution_owned;
  return true;
}

void AdaptiveHeat::output() const
{
  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution, "solution");

  // Add vector for parallel partition.
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::string output_file_name = "output-mesh";

  data_out.write_vtu_with_pvtu_record(/* folder = */ "./",
                                      /* basename = */ output_file_name,
                                      /* index = */ timestep_number,
                                      MPI_COMM_WORLD);
  pcout << "Output written to " << output_file_name << std::endl;
  pcout << "===============================================" << std::endl;
 

}

void AdaptiveHeat::run()
{
  const unsigned int initial_global_refinement = 6;

  //Checkpoint for rollback
  TrilinosWrappers::MPI::Vector old_solution;

  // Setup initial conditions.
  {
    setup();

    VectorTools::interpolate(dof_handler, Functions::ZeroFunction<dim>(), solution_owned);
    solution = solution_owned;


    old_solution.reinit(solution_owned);
  
    time            = 0.00;
    timestep_number = 1;
    


    // Output initial condition.
    output();
  }

  // for time error
  double tol=1e-2;
  double dt_min = 1e-6;
  double dt_max = 1e-1;
    
  std::cout<<"TOL = "<<tol<<std::endl;

  pcout << "===============================================" << std::endl;

  // Time-stepping loop.
  while (time < T - 0.5 * delta_t)
    {
      //Saving previous solution for error estimation
      old_solution = solution_owned;
      
      pcout << "Timestep " << std::setw(3) << timestep_number
            << ", time = " << std::setw(4) << std::fixed << std::setprecision(2)
            << time << " : ";

      double t_old = time;
      double  t_attempt = time + delta_t;
      time = t_attempt; 

      assemble();
      solve_time_step();
    
      // Errore estimation
      TrilinosWrappers::MPI::Vector diff = solution_owned;
      diff.add(-1.0, old_solution); 

      double delta_U = diff.linfty_norm();
      
      pcout << " | Time Var Max: " << std::scientific << delta_U 
            << " | dt: " << delta_t << std::endl;

      // danger for by 0 division
      double denom = std::max(delta_U, 1e-14);
      //Factor clamp
      double factor = std::max(0.3, std::min(tol / denom, 2.0));
      
      // Adaptive Rollback
        if (delta_U > tol) {
            //Rejected step
            pcout << "Exceeded Tolerance (" << delta_U << "), Reducing Timestep..." << std::endl;
            time =t_old;
            
            delta_t = 0.9 * delta_t * factor;
            //Delta_t clamp
            delta_t = std::max(dt_min, std::min(delta_t, dt_max));

            solution_owned = old_solution; // Back to previous solution

            continue;        
        } 
        else {
            //Accepted step
            time = t_attempt;
            timestep_number++;
            
            delta_t = 0.9 * delta_t * factor;
            //Delta_t clamp
            delta_t = std::max(dt_min, std::min(delta_t, dt_max));

            solution = solution_owned;

            output();
        }
      
      if (time < T - 0.5 * delta_t && timestep_number - last_refine_step >= min_steps_between_refine)
      {
        pcout << "-----------------------------------------------" << std::endl;
        pcout << "Applying spatial error-driven refinement" << std::endl;
        
        if(refine_grid(initial_global_refinement, initial_global_refinement + 2)){
            last_refine_step = timestep_number;  // update last refinement 
            pcout << "-----------------------------------------------" << std::endl;
            old_solution.reinit(solution_owned);
            old_solution = solution_owned;
            delta_t = 0.5 * delta_t;
          }
        
      }
    }

}