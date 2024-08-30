
def rank_print(rank, msg):
    if rank == 0:
        print(msg)


def get_toggle_index(num_classes):
    if num_classes == 20:
        toggle_index = [[15, 17, 19], [14, 16, 18]]
    elif num_classes == 24:
        toggle_index = [[23, 22, 20], [14, 21, 19]]
    else:
        toggle_index = [[], []]
    return toggle_index

