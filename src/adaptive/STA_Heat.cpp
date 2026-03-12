#include "STA_Heat.hpp"

void AdaptiveHeat::setup()
{
  pcout << "===============================================" << std::endl;

  // Create the mesh
  {
    pcout << "Initializing the mesh" << std::endl;

    // Read serial mesh.
    Triangulation<dim> mesh_serial;

    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(mesh_serial);

      std::ifstream mesh_file(mesh_file_name);
      grid_in.read_msh(mesh_file);
      Triangulation<dim> mesh_serial(Triangulation<dim>::MeshSmoothing(
                                      Triangulation<dim>::smoothing_on_refinement |
                                      Triangulation<dim>::smoothing_on_coarsening
                                    )
                                  );
    }
    
    // Copy the serial mesh into the parallel one.
    {
      GridTools::partition_triangulation(mpi_size, mesh_serial);

      const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
      mesh.create_triangulation(construction_data);
    }

    pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;

  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;
  
  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);


    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
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

}

void AdaptiveHeat::solve()
{

}

void AdaptiveHeat::refine_grid()
{

}

void AdaptiveHeat::output() const
{

}

void AdaptiveHeat::run()
{

}