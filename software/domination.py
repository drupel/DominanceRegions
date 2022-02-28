class DominanceOrder(SageObject):
    r"""
    A class to compute the dominace order as defined in `arXiv:1902.09507`
    """

    def __init__(self, B):
        self.B = copy(B)
        self.A = ClusterAlgebra(B, principal_coefficients=True)
        self.Aop = ClusterAlgebra(-B, principal_coefficients=True)

    def paths_up_to_length(self, k):
        paths = [ s.path_from_initial_seed() for s in self.A.seeds(mutating_F=False, depth=k) ]
        prefixes = [ p[:-1] for p in paths ]
        return [ p for p in paths if p not in prefixes ]

    @cached_method(key=lambda self, g, depth: (tuple(g), depth))
    def dominated_polytope(self, g, depth):
        paths = self.paths_up_to_length(depth)
        p = paths.pop()
        polytope = cut_along_sequence(g,self.B,p)
        while paths:
            p = paths.pop()
            polytope = polytope.intersection(cut_along_sequence(g,self.B,p))
        return polytope

    def dominated_g_vectors(self, g, depth):
        polytope = self.dominated_polytope(g, depth)
        pts = polytope.integral_points()
        lattice = self.B.transpose().image()
        g = vector(g)
        return [ p for p in pts if p-g in lattice ]

    def show_domination(self, g, depth):
        g = vector(g)
        pts = self.dominated_g_vectors(g, depth)
        polytope = self.dominated_polytope(g, depth)
        return polytope.plot(zorder=-1) + list_plot(pts, color='red', size=50) 

    def generic_F_polynomial(self, g):
        dominated = self.dominated_g_vectors(g, 10)
        dominated.remove(vector(g))
        u = self.A._U.gens()
        return self.A.theta_basis_F_polynomial(g)+sum( var('r%d'%i) * self.A.theta_basis_F_polynomial(h) * u[0]**((g[1]-h[1])/self.B[0,1]) * u[1]**((g[0]-h[0])/self.B[1,0]) for (i,h) in enumerate(dominated)) 

    def reorder(self, f):
        u = self.A._U.gens()
        if self.B[0,1]<0:
            u = list(reversed(u))
        coefficients = f.collect(u[0]).coefficients(u[0])
        collect = list(map(lambda x: [x[0].collect(u[1]),x[1]], coefficients))
        return sum( x[0]*u[0]**x[1] for x in collect)  

    def plot_F_polynomial(self, f):
        u = self.A._U.gens()
        vrs = f.variables()[:-2]
        R = QQ[vrs][u]
        f = SR(f).polynomial(ring=R)
        G = Graphics()
        for p,cf in f.dict().items():
            G += text(cf, p)
        G.axes(show=False)
        return G

    def coxeter_matrix(self):
        return -self.A.euler_matrix()*self.Aop.euler_matrix().inverse()

    def eigenvectors(self):
        return self.coxeter_matrix().eigenvectors_right()

    def opposite_g_vectors(self, g):
        g = vector(g)
        return [ (1+self.B*self.A.euler_matrix().inverse())*g, (1-self.B*self.Aop.euler_matrix().inverse())*g ]

    def find_opposit_corner(self, g):
        g = vector(g)
        C = 2-self.B.apply_map(abs)
        return C.adjugate().transpose()*g
            



def cut_along_sequence(g, B, seq):
    r"""
    Return the intersection of the cones dominated by all the translates of g along the sequence of mutations seq
    """
    current_cone = Polyhedron(rays=B.columns(),base_ring=QQ).translation(g)
    if seq == []:
        return current_cone
    k = seq.pop()
    n = B.ncols()
    Hp = Polyhedron(ieqs=[(0,)*(k+1)+(1,)+(0,)*(n-k-1)])
    Mp = matrix(n, lambda i,j: (1 if i == j else 0) if j != k else (max(B[i,k],0) if i != k else -1) )
    Hm = Polyhedron(ieqs=[(0,)*(k+1)+(-1,)+(0,)*(n-k-1)])
    Mm = matrix(n, lambda i,j: (1 if i == j else 0) if j != k else (max(-B[i,k],0) if i != k else -1) )
    new_g = (Mp if g in Hp else Mm) * vector(g)
    new_B = copy(B)
    new_B.mutate(k)
    polytope = cut_along_sequence(new_g, new_B, seq)
    polytope_p = Mm*(polytope.intersection(Hp))
    polytope_m = Mp*(polytope.intersection(Hm))
    polytope = polytope_p.convex_hull(polytope_m)
    return polytope.intersection(current_cone)

def region_vertices(b,c,g):
    #this is only for rank 2
    vertices = []
    vertices.append(g)
    vertices.append(vector((0,(b*c+sqrt(b*c*(b*c-4)))*g[0]/(2*b)+g[1])))
    vertices.append(vector(((b/sqrt(b*c*(b*c-4)))*((b*c+sqrt(b*c*(b*c-4)))*g[0]/b+(b*c+sqrt(b*c*(b*c-4)))*g[1]/2),(c/sqrt(b*c*(b*c-4)))*((b*c+sqrt(b*c*(b*c-4)))*g[1]/c+(b*c+sqrt(b*c*(b*c-4)))*g[0]/2))))
    vertices.append(vector(((b*c+sqrt(b*c*(b*c-4)))*g[1]/(2*c)+g[0],0)))
    return vertices


def foo(g, B, seq):
    r"""
    Return the intersection of the cones dominated by all the translates of g along the sequence of mutations seq
    """
    current_cone = Polyhedron(rays=B.columns(),base_ring=QQ).translation(g)
    if seq == []:
        return current_cone
    k = seq.pop()
    n = B.ncols()
    Hp = Polyhedron(ieqs=[(0,)*(k+1)+(1,)+(0,)*(n-k-1)])
    Mp = matrix(n, lambda i,j: (1 if i == j else 0) if j != k else (max(B[i,k],0) if i != k else -1) )
    Hm = Polyhedron(ieqs=[(0,)*(k+1)+(-1,)+(0,)*(n-k-1)])
    Mm = matrix(n, lambda i,j: (1 if i == j else 0) if j != k else (max(-B[i,k],0) if i != k else -1) )
    new_g = (Mp if g in Hp else Mm) * vector(g)
    new_B = copy(B)
    new_B.mutate(k)
    polytope = foo(new_g, new_B, seq)
    polytope_p = Mm*(polytope.intersection(Hp))
    polytope_m = Mp*(polytope.intersection(Hm))
    polytope = polytope_p.convex_hull(polytope_m)
    return polytope #.intersection(current_cone)
