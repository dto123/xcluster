from deltasep_utils import gen_k_centers


def main():
    centers, delta = gen_k_centers(3, 2)
    print "hi"
    print centers
    return centers
    return delta

main()
