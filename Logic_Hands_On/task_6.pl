connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

can_move(X,Y) :-
    connected(X,Y).

can_move(X,Z) :-
	connected(X,Y),
	can_move(Y,Z). % fixed for transitivity


valid_move(X,Y) :-
	connected(X,Y).
%
% (a) Why does Prolog return true for can move(a,b)?
% Because it is given as a predicate.
% can_move(a,b) :- connected(a,b)
% since connected(a,b), can_move(a,b)
%
% (b) Why does it not establish can move(a,c)?
% Because can_move(a,c) can not infer connected(a,c)
% There is no line to add new predicates, or call the predicate recursively.
%
% (c) What is the relationship between the Prolog rule can move(X,Y) and the logical
% implication
% Connected(X, Y ) → CanMove(X, Y )?

% The prolog rule is equivalent to the given implication