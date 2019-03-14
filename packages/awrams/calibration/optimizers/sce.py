from awrams.calibration.optimizer import Optimizer
from awrams.utils.messaging import Message
import numpy as np
from awrams.calibration.parameters import ParameterPopulation, ParameterPoint

#Signed int32 maximum value
RAND_MAX = 2147483647

class ShuffledOptimizer(Optimizer):
    def __init__(self,pspace,eval_fac,evolver_spec,n_complexes,complex_size=None,min_complexes=None,reduce_wait=0,max_eval=20000,max_nsni=5,min_imp=0.01,
        init_method='uniform',random_shuffle=False,seed=0):
        super().__init__()
        self.pspace = pspace # ParameterSpace
        self.eval_fac = eval_fac # Factory class of evaluator
        if complex_size is None:
            complex_size = len(pspace) * 2 + 1
        self.complex_size = complex_size # Members per complex
        self.n_complexes = n_complexes # (Initial) number of complexes
        if min_complexes is None:
            min_complexes = 1
        self.min_complexes = min_complexes # Minimum number of complexes
        self.reduce_wait = reduce_wait # How many shuffles to wait before reducing complexes
        self.evolver_spec = evolver_spec # Specification of evolver
        self.max_eval = max_eval # Maximum model evaluations before temrinating
        self.max_nsni = max_nsni # Maximum number of shuffles without improvement
        self.min_imp = min_imp # Minimum proportion of improvement to occur over max_nsni
        self.init_method = init_method # String key for creting initial population (member of 'init_methods' dict)
        self.seed = seed # Seed used by RNG
        self.random_shuffle = random_shuffle # Boolean for whether shuffling is random or ordered by score

    def run(self):
        try:
            np.random.seed(self.seed)
            self.rng = np.random.RandomState(np.random.randint(0,RAND_MAX)) #Maximum value as per randint docs

            pop_eval = self.eval_fac.new()

            self.evolvers = self.build_remote()

            initial_params = init_methods[self.init_method](self.pspace,self.n_complexes * self.complex_size)
            
            self.status = Message('evaluating initial population')
            self.update_status()

            score = pop_eval.evaluate_population(initial_params)

            self.population = ParameterPopulation(score,initial_params)
            self.population.sort()

            self.prev_best = [self.population.score.min()]

            self.n_shuffle = 0
            self.n_eval = len(self.population)

            if self.random_shuffle:
                shuffler = self.rng
            else:
                shuffler = None

            

            convergence = self.check_convergence()

            from concurrent.futures import ThreadPoolExecutor
            self.evolve_exe = ThreadPoolExecutor(self.n_complexes)

            while convergence is None:
                
                self.status = Message('running',n_eval=self.n_eval,n_shuffle = self.n_shuffle, best_params=self.population[0])

                self.check_status()

                complexes = assign_to_complexes(self.population,self.n_complexes,shuffler)

                self.population, n_eval = self.evolve(complexes)
                self.n_eval += n_eval
                self.n_shuffle += 1

                self.population.sort()

                convergence = self.check_convergence()

                self.reduce_complexes()

            self.status = Message('complete',condition=convergence,n_eval=self.n_eval,best_params=self.population[0])

            self.update_status()
        
        except Exception as e:
            print(e)
            raise

    def check_convergence(self):
        if (self.n_eval >= self.max_eval):
            return 'Maximum evaluations completed'

        cur_best = self.population.score.min()

        if len(self.prev_best) >= self.max_nsni: # +++ Seem to be miscounting this, BUG
            imp_target = self.prev_best[0] - (self.prev_best[0] * self.min_imp)
            if cur_best >= imp_target:
                return "Maximum non-improving shuffle loops executed"
            else:
                self.prev_best = []

        self.prev_best.append(cur_best)
        if len(self.prev_best) > self.max_nsni:
            self.prev_best.pop(0)

    def reduce_complexes(self):
        if self.n_shuffle > self.reduce_wait:
            if self.n_complexes > self.min_complexes:
                if self.n_complexes > 1:
                    self.population = self.population[:-self.complex_size]
                    self.n_complexes -= 1

    def evolve(self,complexes):
        futures = []
        for i,cplex in enumerate(complexes):
            futures.append(self.evolve_exe.submit(self.evolvers[i].run,cplex,**self.evolver_spec.evolver_run_args))

        results = [f.result() for f in futures]

        n_eval = sum([r[1] for r in results])
        new_pop = ParameterPopulation.join_populations([r[0] for r in results])

        return new_pop,n_eval

    def build_remote(self):
        #eval_to_remote = Queue()
        #remote_to_eval = [Queue() for i in range(ncomplexes)]
        evolvers = []
        
        for i in range(self.n_complexes):
            rng = np.random.RandomState(self.rng.randint(0,RAND_MAX))
            evaluator = self.eval_fac.new()
            evolvers.append(self.evolver_spec.evolver_cls(self.pspace,evaluator,rng,**self.evolver_spec.evolver_init_args))
            
        return evolvers

    def get_max_evaluators(self):
        return self.n_complexes + 1

def assign_to_complexes(population,n_complexes,rng=None):
    
    if rng is None:
        population.sort()
    else:
        population.shuffle(rng)
    population_c = [[] for i in range(n_complexes)]

    for i,p in enumerate(population):
        ccomplex = i%n_complexes
        population_c[ccomplex].append(p)
    
    complexes = []
    for c in population_c:
        complexes.append(ParameterPopulation.from_points(c))
    
    return complexes


class CCEvolver:
    def __init__(self,pspace,evaluator,rng):
        self.pspace = pspace
        self.evaluator = evaluator
        self.rng = rng
        self.simplex_size = len(pspace) + 1

    def run(self,population,n_offspring,n_evolutions,alpha=1.0,beta=0.5,elitism=1.0,rscale=2.0):
        n_eval = 0
        
        for e in range(n_evolutions):
            sidx = get_simplex(self.simplex_size,len(population),elitism,self.rng)
            
            simplex = population[sidx]
            
            simplex = simplex.sort()
            
            for o in range(n_offspring):
                #Identify worst point, centroid of others
                centroid = simplex[:-1].params.mean(axis=0)
                
                worst_p = simplex[-1].params
                best_p = simplex[0].params
                
                #Reflection
                
                ref_p = centroid + alpha*(centroid-worst_p)
                if out_of_bounds(ref_p,self.pspace):
                    ref_p = random_normal_point(self.pspace,self.rng,best_p,population.params.std(axis=0)+1.e-99)
                    #ref_p = random_normal_point(self.pspace,self.rng,population.params.mean(axis=0),rscale * population.params.std(axis=0))
                        
                
                score = self.evaluator.evaluate(ref_p)
                n_eval += 1
                
                if score < simplex[-1].score:
                    simplex[-1] = ParameterPoint(score,ref_p)
                else:
                    #Contraction
                    con_p = worst_p + beta *(centroid-worst_p)
                    score = self.evaluator.evaluate(con_p)
                    n_eval += 1
                          
                    if score < simplex[-1].score:
                        simplex[-1] = ParameterPoint(score,con_p)
                    else:
                        #new_p = self.rng.normal(population.params.mean(axis=0),rscale * population.params.std(axis=0))
                        #new_p = random_normal_point(self.pspace,self.rng,population.params.mean(axis=0),rscale * population.params.std(axis=0))
                        new_p = random_normal_point(self.pspace,self.rng,best_p,population.params.std(axis=0)+1.e-99)
                        score = self.evaluator.evaluate(new_p)
                        
                        n_eval += 1
                        
                        simplex[-1] = ParameterPoint(score,new_p)
                
                simplex.sort()
                
            population[sidx] = simplex
            population.sort()
            
        return population, n_eval

def random_normal_point(pspace,rng,means,sds):
    pt = rng.normal(means,sds)
    for i,p in enumerate(pt):
        valid = not (p < pspace.min[i] or p > pspace.max[i])
        while not valid:
            pt[i] = p = rng.normal(means[i],sds[i])
            valid = not (p < pspace.min[i] or p > pspace.max[i])
    return pt




def init_lhs(pspace,n):
    from pyDOE import lhs
    population = pspace.range * lhs(len(pspace),n) + pspace.min
    return population

def init_uniform(pspace,n):
    population = pspace.range * np.random.uniform(size=(n,len(pspace))) + pspace.min
    return population    

init_methods = {
    'lhs': init_lhs,
    'uniform': init_uniform

}

def triangular(npts,elitism=0.0,rng=None):
    if rng is None:
        rng = np.random
    x = rng.uniform(size=npts)
    return x ** (1.0 + elitism)

def get_simplex(npts,popsize,elitism=0.0,rng=None):
    s = triangular(npts,elitism,rng)
    avail = list(range(popsize))
    simplex = []
    for x in s:
        x = int(np.floor(x*len(avail)))
        simplex.append(avail.pop(x))
    return simplex
    
def constrain(v,min,max):
    return np.min([np.max([v,min]),max])

def constrain_params(pset,pspace):
    return np.array([constrain(p,pspace.min[i],pspace.max[i])\
                     for i,p in enumerate(pset)])

def out_of_bounds(pset,pspace):
    return (pset < pspace.min).any() or (pset >pspace.max).any()