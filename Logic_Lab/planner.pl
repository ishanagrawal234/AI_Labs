% =====================================================================
% Artificial Intelligence Laboratory -- Logical Planning
% Task 8, 9, 10: Prolog as a Plan Verifier and Rule-Based Reasoner
% =====================================================================

% --- Task 8: Warehouse Topology Facts ---
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

% Rule: The robot can move between connected locations
can_move(X, Y) :-
    connected(X, Y).

% --- Task 9: Plan Step Verification ---
valid_move(X, Y) :-
    connected(X, Y).

% Sequential plan verification for a list of moves
valid_plan([move(X, Y)]) :-
    valid_move(X, Y).
valid_plan([move(X, Y) | Rest]) :-
    valid_move(X, Y),
    valid_plan(Rest).

% --- Task 10: Rule-Based Logical Inference Chain ---
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.

