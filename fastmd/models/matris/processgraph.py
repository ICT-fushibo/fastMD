import torch
from fastmd.graph import RadiusGraph
from fastmd.graph.geometry import small_matmul
from collections.abc import Sequence
from fastmd.models.matris.functions import segment_count

    
def process_graphs(graphs: Sequence[RadiusGraph], 
                   compute_stress: bool = True):
    """
    Process a sequence of graphs and batch them into a unified graph.
    
    Args:
        graphs: Sequence of RadiusGraph objects containing multiple crystal graph data.
        compute_stress: Whether to compute stress.
    
    Returns:
        batched_graph:
            atomic_numbers: Atomic numbers of all atoms [total_atoms]
            edge_lengths: Lengths of all edges [total_direct_edges]
            unit_edge_vectors: Unit edge vectors [total_direct_edges, 3]
            atom_segment: Graph(Batch) indices for each atom [total_atoms]
            atoms_per_graph: List of atom counts per graph
            directed2undirected: Mapping from directed to undirected edges [total_direct_edges]
            undirected2directed: Mapping from undirected to directed edges [total_undirected_edges]
            volumes: Volume of each graph [num_graphs, 1, 1]
            lattice: Lattice vectors [num_graphs*3, 3]
            atom_active_mask: Whether each atom has at least one graph edge
                [total_atoms]
            
            batch_cart_coords: Cartesian coordinates of all atoms [total_atoms, 3]
            batch_strains: Strain tensors [num_graphs, 3, 3] (only when compute_stress=True)
            
            atom_graph_dict: Dict containing atom graph information
                atom_graph: Atom connectivity [total_direct_edges, 2]
                target_index/source_index: Target/source node indices for edges [total_direct_edges]
                num_segment: Number of segments
                source_bincount: The frequency of each source node in atom graph
                target_bincount: The frequency of each target node in atom graph
            line_graph_dict: Dict containing line graph information
                line_graph: Compressed line graph [num_angles, 3] # remove directed edge index
                atom_list: Central atom indices for angles [num_angles]
                target_index/source_index: Target/source edge indices for line graph [num_angles]
                target_DE_index/source_DE_index: Directed edge indices [num_angles]
                source_bincount: The frequency of each source node in line graph
                target_bincount: The frequency of each target node in line graph
    """
    
    batched_graph = {}
    num_graphs = len(graphs)
    device = graphs[0].lattice.device
    dtype = graphs[0].lattice.dtype
    
    atomic_numbers = []
    batched_atom_graph, batched_line_graph = [], []
    atom_segment, atoms_per_graph = [], []
    
    batch_lattice, batch_image, batch_cart_coords = [], [], []
    undirected2directed, directed2undirected = [], []
    
    atom_graph_len, num_undirected2directed, num_directed2undirected = 0, 0, 0
    for graph_idx, graph in enumerate(graphs):
        direct_edge_num = len(graph.atom_graph)
        undirected2directed.append(graph.undirected2directed)
        num_undirected2directed += int(direct_edge_num/2) # undirect edge num
         
        directed2undirected.append(graph.directed2undirected)
        num_directed2undirected += direct_edge_num # direct edge num 
        
        batched_atom_graph.append(graph.atom_graph)
        atom_graph_len += direct_edge_num # direct edge num  

    # Allocate offset buffers directly on device (avoids per-step H2D .to(device)).
    atom_offset_idx_list = torch.empty((atom_graph_len, 2), dtype=torch.int32, device=device)
    edge_offses_idx_list = torch.empty(num_undirected2directed, dtype=torch.int32, device=device)
    n_undirected_list = torch.empty(num_directed2undirected, dtype=torch.int32, device=device)

    atom_idx_offset, edge_idx_offset, undirected_list_offset, directed_list_offset = 0, 0, 0, 0
    start_atom_offset, start_edge_offset, start_n_undirected = 0, 0, 0
    
    for graph_idx, graph in enumerate(graphs):
        atomic_numbers.append(graph.atomic_number) # atom type
        n_atom = graph.atomic_number.shape[0]
        atoms_per_graph.append(n_atom)
        if graph.atom_frac_coord.requires_grad:
            #graph.atom_frac_coord.requires_grad = False
            graph.atom_frac_coord = graph.atom_frac_coord.detach()
        if graph.lattice.requires_grad:
            #graph.lattice.requires_grad = False
            graph.lattice = graph.lattice.detach()

        # Geometry products use small_matmul: cuBLAS would run them in TF32
        # whenever allow_tf32 is set (the default under the NGC containers'
        # TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1), ~1e-2 A.
        atom_cart_coords = small_matmul(graph.atom_frac_coord, graph.lattice)  # [n_atom, 3]
        
        batch_cart_coords.append(atom_cart_coords)
        batch_lattice.append(graph.lattice)
        batch_image.append(graph.neighbor_image)

        edge_offses_idx_list[start_edge_offset : start_edge_offset+len(graph.undirected2directed)] = edge_idx_offset
        atom_offset_idx_list[start_atom_offset : start_atom_offset+len(graph.atom_graph), :] = atom_idx_offset  # scalar fill (no H2D)
        n_undirected_list[start_n_undirected : start_n_undirected+len(graph.directed2undirected)] = undirected_list_offset

        if len(graph.line_graph) != 0:
            this_line_graph = graph.line_graph.new_zeros([graph.line_graph.shape[0], 5])
            this_line_graph[:, 0] = graph.line_graph[:, 0] + atom_idx_offset # atom type at this angle
            this_line_graph[:, 1] = graph.line_graph[:, 1] + undirected_list_offset # left undirect edge index
            this_line_graph[:, 3] = graph.line_graph[:, 3] + undirected_list_offset # right undirect edge index
            
            this_line_graph[:, 2] = graph.line_graph[:, 2] + directed_list_offset # left direct edge index 
            this_line_graph[:, 4] = graph.line_graph[:, 4] + directed_list_offset # right direct edge index  
            
            batched_line_graph.append(this_line_graph)
        
        atom_segment.append(torch.full((n_atom,), graph_idx, dtype=torch.int64, device=device))
        atom_idx_offset += n_atom
        edge_idx_offset += len(graph.atom_graph)
        directed_list_offset += len(graph.directed2undirected)
        undirected_list_offset += len(graph.undirected2directed)

        start_edge_offset += len(graph.undirected2directed)
        start_atom_offset += len(graph.atom_graph)
        start_n_undirected += len(graph.directed2undirected)
    
    # (offset buffers + atom_segment already built on device)


    # ======= packing features =======
    atomic_numbers = torch.cat(atomic_numbers, dim=0)
    batched_atom_graph = torch.cat(batched_atom_graph, dim=0) # [edge, 2]
    batched_atom_graph += atom_offset_idx_list #add offset
    batched_atom_graph = batched_atom_graph.to(torch.int64)
    
    if batched_line_graph != []:
        batched_line_graph = torch.cat(batched_line_graph, dim=0) # line_graph
        batched_line_graph = batched_line_graph.to(torch.int64)
    else:  # when line graph is empty
        batched_line_graph = torch.tensor([]) 

    atom_segment = (
        torch.cat(atom_segment, dim=0).type(torch.int64)
    )
    
    directed2undirected = torch.cat(directed2undirected, dim=0)
    directed2undirected += n_undirected_list #add offset
    undirected2directed = torch.cat(undirected2directed, dim=0)
    undirected2directed += edge_offses_idx_list # add offset
    
    batch_image = torch.cat(batch_image, dim=0)  # [direct_edge_num, 3]
    batch_lattice = torch.cat(batch_lattice, dim=0) #[num_graphs * 3, 3]
    batch_cart_coords = torch.cat(batch_cart_coords, dim=0)
    
    # Reshape [n_graphs*3, 3] -> [n_graphs, 3, 3] for volumes calculation
    batch_lattices = batch_lattice.reshape(num_graphs, 3, 3)
    cross_product = torch.linalg.cross(batch_lattices[:, 1], batch_lattices[:, 2])
    volumes = (batch_lattices[:, 0] * cross_product).sum(dim=-1)
    volumes = volumes.unsqueeze(1).unsqueeze(2)
    
    batch_line_graph_compress = torch.tensor([]) 
    if len(batched_line_graph) != 0:
        # line graph index
        line_graph_atom_index = batched_line_graph[:, 0]
        line_graph_UDE_target_index = batched_line_graph[:, 1]
        line_graph_DE_target_index = batched_line_graph[:, 2]
        line_graph_UDE_source_index = batched_line_graph[:, 3]
        line_graph_DE_source_index = batched_line_graph[:, 4]
        
        batch_line_graph_compress = batched_line_graph.new_zeros([batched_line_graph.shape[0], 3])
        batch_line_graph_compress[:, 0] = line_graph_atom_index
        batch_line_graph_compress[:, 1] = line_graph_UDE_target_index 
        batch_line_graph_compress[:, 2] = line_graph_UDE_source_index

    #========================================
    strains = None
    batch_cart_coords.requires_grad_(True) 
    if compute_stress:
        strains = torch.zeros(
            (num_graphs, 3, 3),
            dtype=dtype,
            device=device,
        )

        strains.requires_grad_(True)
        symmetric_strains = 0.5 * (
            strains + strains.transpose(-1, -2)
        )

        if len(set(atoms_per_graph)) == 1:
            atoms_per_system = atoms_per_graph[0]
            coords_by_graph = batch_cart_coords.view(
                num_graphs, atoms_per_system, 3
            )
            batch_cart_coords = (
                coords_by_graph
                + small_matmul(coords_by_graph, symmetric_strains)
            ).view(-1, 3)
        else:
            coord_chunks = torch.split(batch_cart_coords, atoms_per_graph)
            batch_cart_coords = torch.cat(
                [
                    coords + small_matmul(coords, symmetric_strains[graph_idx])
                    for graph_idx, coords in enumerate(coord_chunks)
                ],
                dim=0,
            )
        batch_lattice = batch_lattice.view(-1, 3, 3)
        batch_lattice = batch_lattice + small_matmul(batch_lattice, symmetric_strains)
        batch_lattice = batch_lattice.view(-1, 3)
    #========================================
    
    # node index
    target_index, source_index  = batched_atom_graph[:, 0], batched_atom_graph[:, 1]
    # ===== compute pairwise distance and edge vectors =====
    center_pos = batch_cart_coords[target_index] 
    neighbor_pos = batch_cart_coords[source_index]
    if num_graphs == 1:
        image_cart = small_matmul(batch_image, batch_lattice)
    elif len({graph.atom_graph.shape[0] for graph in graphs}) == 1:
        edges_per_system = graphs[0].atom_graph.shape[0]
        image_cart = small_matmul(
            batch_image.view(num_graphs, edges_per_system, 3),
            batch_lattice.view(num_graphs, 3, 3),
        ).view(-1, 3)
    else:
        image_chunks = torch.split(
            batch_image,
            [graph.atom_graph.shape[0] for graph in graphs],
        )
        lattices = batch_lattice.view(num_graphs, 3, 3)
        image_cart = torch.cat(
            [
                small_matmul(image, lattices[graph_idx])
                for graph_idx, image in enumerate(image_chunks)
            ],
            dim=0,
        )
    neighbor_pos = neighbor_pos + image_cart
    edge_vectors = center_pos - neighbor_pos
    edge_lengths = torch.norm(edge_vectors, dim=1) # pairwise distance
    unit_edge_vectors = edge_vectors / edge_lengths[:, None] # edge vectors
    
        
    batched_graph['atomic_numbers'] = atomic_numbers # [atoms]
    batched_graph['edge_lengths'] = edge_lengths # [direct_edge_num]
    batched_graph['unit_edge_vectors'] = unit_edge_vectors # [direct_edge_num, 3]
    batched_graph['atom_segment'] = atom_segment.to(device=device) # [atoms]
    batched_graph['atoms_per_graph'] = atoms_per_graph # List, size:[atoms]
    batched_graph['num_graphs'] = num_graphs # host int (no sync)
    batched_graph['directed2undirected'] = directed2undirected.to(torch.int64) #[direct_edge_num]
    batched_graph['undirected2directed'] = undirected2directed.to(torch.int64) #[undirect_edge_num]
    batched_graph['volumes'] = volumes # [num_graphs, 1, 1]
    batched_graph['lattice'] = batch_lattice # [num_graphs*3, 3]
    batched_graph['batch_cart_coords'] = batch_cart_coords # [atoms, 3]
    batched_graph['batch_strains'] = strains # [num_graphs, 3, 3]
    
    atom_graph_dict, line_graph_dict = {}, {}
    atom_graph_dict['atom_graph'] = batched_atom_graph # [direct_edge_num, 2]
    atom_graph_dict['target_index'] = target_index # [direct_edge_num]
    atom_graph_dict['source_index'] = source_index # [direct_edge_num]
    # Host-known segment counts -> pass minlength to torch.bincount so it does
    # NOT read max() on the host (a D2H sync). total_atoms / num_undirected are
    # plain Python ints from tensor shapes (no sync).
    total_atoms = atomic_numbers.shape[0]
    num_undirected = undirected2directed.shape[0]
    atom_graph_dict['num_segment'] = total_atoms
    atom_graph_dict['num_undirected'] = num_undirected

    bincount_source_atom_graph = segment_count(source_index, total_atoms)
    bincount_source_atom_graph = bincount_source_atom_graph.where(bincount_source_atom_graph != 0, bincount_source_atom_graph.new_ones(1))
    target_count_atom_graph = segment_count(target_index, total_atoms)
    # Retain zero-degree information before clamping the attention counts.
    # This fixed-shape device mask also works during CUDA Graph replay as
    # atoms disconnect/reconnect, without deleting nodes or host synchronization.
    batched_graph['atom_active_mask'] = target_count_atom_graph.ne(0)
    if all(getattr(graph, "atom_target_sorted", False) for graph in graphs):
        atom_graph_dict['_target_segment_attention_counts'] = target_count_atom_graph
    bincount_target_atom_graph = target_count_atom_graph.where(target_count_atom_graph != 0, target_count_atom_graph.new_ones(1))
    atom_graph_dict['source_bincount'] = bincount_source_atom_graph
    atom_graph_dict['target_bincount'] = bincount_target_atom_graph
    
    line_graph_dict['line_graph'] = batch_line_graph_compress
    if len(line_graph_dict['line_graph']) != 0:
        line_graph_dict['atom_list'] = line_graph_atom_index # [angles, 3]
        line_graph_dict['target_index'] = line_graph_UDE_target_index  # [angles]
        line_graph_dict['source_index'] = line_graph_UDE_source_index  # [angles]
        line_graph_dict['target_DE_index'] = line_graph_DE_target_index # [angles]
        line_graph_dict['source_DE_index'] = line_graph_DE_source_index # [angles]
        
        bincount_source_line_graph = segment_count(line_graph_UDE_source_index, num_undirected)
        bincount_source_line_graph = bincount_source_line_graph.where(bincount_source_line_graph != 0, bincount_source_line_graph.new_ones(1))
        bincount_target_line_graph = segment_count(line_graph_UDE_target_index, num_undirected)
        bincount_target_line_graph = bincount_target_line_graph.where(bincount_target_line_graph != 0, bincount_target_line_graph.new_ones(1))
        line_graph_dict['source_bincount'] = bincount_source_line_graph
        line_graph_dict['target_bincount'] = bincount_target_line_graph 
    
    batched_graph['atom_graph_dict'] = atom_graph_dict
    batched_graph['line_graph_dict'] = line_graph_dict

    # Contract verification and lowering selection happen while the graph is
    # prepared/captured. CUDA Graph replay consumes the already compiled plan.
    from fastmd.models.matris.topology import bind_matris_topology, matris_lowering_requests
    from fastmd.engine.topology_lowering import compile_lowering_plan

    topology_binding = bind_matris_topology(batched_graph, graphs)
    topology_plan = compile_lowering_plan(
        topology_binding.instance.schema,
        topology_binding.instance.certificate,
        matris_lowering_requests(),
    )
    batched_graph['topology_contract'] = topology_binding.instance
    batched_graph['topology_verification'] = topology_binding.report
    batched_graph['topology_lowering_plan'] = topology_plan
    atom_graph_dict['_topology_lowering_plan'] = topology_plan
    line_graph_dict['_topology_lowering_plan'] = topology_plan
    atom_graph_dict['_topology_graph_kind'] = 'atom'
    line_graph_dict['_topology_graph_kind'] = 'line'
    # The cached OpDispatcher of (policy, certified layout) under "_ops": the
    # model sites' only operator path.
    from fastmd.engine.dispatch import attach_dispatcher
    attach_dispatcher(batched_graph, topology_plan)
    
    return batched_graph
