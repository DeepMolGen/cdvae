import argparse
import torch
from pymatgen.core.periodic_table import Element

from cdvae.common.data_utils import frac_to_cart_coords

atomic_number_to_symbol = {element.Z: element.symbol for element in Element}

def main(gen_path, args):
    x = torch.load(gen_path)
    num_atoms = x["num_atoms"]
    atom_types = x["atom_types"]
    frac_coords = x["frac_coords"]
    lengths = x["lengths"]
    angles = x["angles"]
    # convert fractional coordinates to cartesian 
    cart_coords = frac_to_cart_coords(frac_coords[0], lengths[0], angles[0], num_atoms[0])

    index_list = torch.cumsum(num_atoms[0], dim=0).numpy().tolist()
    indice_tuples = []
    for i, ii in enumerate(index_list):
        if i == 0:
            tup = [0, index_list[i] - 1]
        else:
            tup = [index_list[i - 1] - 1, index_list[i] - 1]
        indice_tuples.append(tup)

    #for id_needed in range(num_atoms.shape[1]):
    for id_needed in range(args.num_materials_to_xyz): # output a given number of materials
        id_fracs = frac_coords[0][
            indice_tuples[id_needed][0] : indice_tuples[id_needed][1]
        ]
        id_atom_types = atom_types[0].numpy()[
            indice_tuples[id_needed][0] : indice_tuples[id_needed][1]
        ]
        id_cart_coords = cart_coords.numpy()[
            indice_tuples[id_needed][0] : indice_tuples[id_needed][1]
        ]

        # write xyz file
        f = open(args.output_path + "crystal" + str(id_needed) + ".xyz", "w")
        f.write("%d\n\n" % len(id_atom_types)) # number of atoms
        for i in range(len(id_atom_types)):
            atom_symbol = atomic_number_to_symbol[id_atom_types[i]]
            f.write("%s %.9f %.9f %.9f\n" % (atom_symbol, id_cart_coords[i, 0], id_cart_coords[i, 1], id_cart_coords[i, 2]))
        f.close()

if __name__ == '__main__':
    gen_path = "/Users/luisaorozco/Documents/Projects/DeepMolGen/cdvae/hydra/model/eval_gen.pt"
    out_path = "/Users/luisaorozco/Documents/Projects/DeepMolGen/cdvae/hydra/model/generated_samples/"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_materials_to_xyz', default=1, type=int)
    parser.add_argument('--eval_gen_path', default=gen_path)
    parser.add_argument('--output_path', default=out_path)

    args = parser.parse_args()

    main(gen_path, args = parser.parse_args())