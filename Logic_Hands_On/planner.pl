% Facts: which locations are directly connected
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

% Task 6: basic movement rule
can_move(X,Y) :- connected(X,Y).

% Task 7: verifying a proposed plan step-by-step
valid_move(X,Y) :- connected(X,Y).

% Example queries to run in SWI-Prolog:
%   ?- can_move(a,b).       % true
%   ?- can_move(a,c).       % false
%   ?- valid_move(a,b).     % true
%   ?- valid_move(b,c).     % true
%   ?- valid_move(a,c).     % false  <- catches an invalid Move(a,c)

% Task 8: chained rules
wet_road.
slippery :- wet_road.
reduce_speed :- slippery.
%   ?- reduce_speed.        % true