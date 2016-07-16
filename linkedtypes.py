class resource(object):
    """ This class represents a resource concerning linked data. """

    def __init__(self, resource_id=''):
        """
        Initializes a new resource wit the given resource identifier.
        :param resource_id: the unique identifier of the resource.
        """
        self.resource_id = resource_id

    def __repr__(self):
        return '<%s>' % repr(self.resource_id)

    def __hash__(self):
        return hash(self.resource_id)

    def __eq__(self, other):
        return self.resource_id == other.resource_id if isinstance(other, resource) else False


class triple(tuple):
    """ This class represents a triple concerning linked data. """

    @property
    def subject(self):
        return self[0]

    @property
    def predicate(self):
        return self[1]

    @property
    def object(self):
        return self[2]

    def __repr__(self):
        return '%s %s %s .' % (self.subject, self.predicate, self.object)

    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))

    def __eq__(self, other):
        return self.subject == other.subject and self.predicate == other.predicate and self.object == other.object if isinstance(
            other, triple) else False


class graph(object):
    """ This class represents a graph concerning linked data, containing a set of triple."""

    def __init__(self, triples = None):
        self.triples = frozenset(triples) if triples else frozenset()

    def subjects(self):
        return frozenset(t.subject for t in self.triples)

    def predicates(self):
        return frozenset(t.predicate for t in self.triples)

    def objects(self):
        return frozenset(t.objects for t in self.objects)

    def __iter__(self):
        return iter(self.subjects())

    def __getitem__(self, item):
        if not isinstance(item, resource):
            raise TypeError('The item must be a resource (not "%s").' % item.__class__)
        return subjectgraph([t for t in self.triples if t.subject == item])

    def __add__(self, other):
        if not isinstance(other, graph):
            raise TypeError('Can only concatenate a graph (not "%s") to graph' % other.__class__)
        return graph(self.triples.union(other.triples))

    def __repr__(self):
        return '{ %s }' % ' '.join(map(str, self.triples))


class subjectgraph(graph):
    """ This class represents a graph concerning linked data, containing a set of triple."""

    def __iter__(self):
        return iter(self.predicates())

    def __getitem__(self, item):
        if not isinstance(item, resource):
            raise TypeError('The item must be a resource (not "%s").' % item.__class__)
        return [t.object for t in self.triples if t.predicate == item]

    def __add__(self, other):
        if not isinstance(other, graph):
            raise TypeError('Can only concatenate a graph (not "%s") to graph' % other.__class__)
        return self.triples.union(other.triples)
