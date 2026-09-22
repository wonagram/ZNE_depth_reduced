# ============================================================
# U parameters
# ============================================================
theta_x1 = 0.5
theta_y1 = 0.6
theta_z1 = 0.8
theta_x2 = 0.4
theta_y2 = 0.4
theta_z2 = 0.4
theta_x3 = 0.6
theta_y3 = 0.5
theta_z3 = 0.9
theta_x4 = 0.3
theta_y4 = 0.4
theta_z4 = 0.5
theta_x5 = 0.8
theta_y5 = 0.7
theta_z5 = 0.6
theta_x6 = 0.7
theta_y6 = 0.7
theta_z6 = 0.7

# ============================================================
# U gate sequence
# ============================================================
U_GATES = {
    1: [
        # Layer 1
        [("x", 0, theta_x1)],

        # Layer 2
        [("y", 0, theta_y1)],

        # Layer 3
        [("z", 0, theta_z1)],

        # Layer 4
        [("x", 0, theta_x2)],

        # Layer 5
        [("y", 0, theta_y2)],

        # Layer 6
        [("z", 0, theta_z2)],

        # Layer 7
        [("x", 0, theta_x3)],

        # Layer 8
        [("y", 0, theta_y3)],

        # Layer 9
        [("z", 0, theta_z3)],

        # Layer 10
        [("x", 0, theta_x4)],

        # Layer 11
        [("y", 0, theta_y4)],

        # Layer 12
        [("z", 0, theta_z4)],

        # Layer 13
        [("x", 0, theta_x5)],

        # Layer 14
        [("y", 0, theta_y5)],

        # Layer 15
        [("z", 0, theta_z5)],

        # Layer 16
        [("x", 0, theta_x6)],

        # Layer 17
        [("y", 0, theta_y6)],

        # Layer 18
        [("z", 0, theta_z6)],
    ],

    2: [
        # Layer 1
        [
            ("x", 0, theta_x1),
            ("x", 1, theta_x1),
        ],

        # Layer 2
        [
            ("cx", 0, 1),
        ],

        # Layer 3
        [
            ("y", 0, theta_y1),
            ("y", 1, theta_y1),
        ],

        # Layer 4
        [
            ("cx", 1, 0),
        ],

        # Layer 5
        [
            ("z", 0, theta_z1),
            ("z", 1, theta_z1),
        ],

        # Layer 6
        [
            ("cx", 0, 1),
        ],

        # Layer 7
        [
            ("x", 0, theta_x2),
            ("x", 1, theta_x2),
        ],

        # Layer 8
        [
            ("cx", 1, 0),
        ],

        # Layer 9
        [
            ("y", 0, theta_y2),
            ("y", 1, theta_y2),
        ],

        # Layer 10
        [
            ("cx", 0, 1),
        ],

        # Layer 11
        [
            ("z", 0, theta_z2),
            ("z", 1, theta_z2),
        ],

        # Layer 12
        [
            ("cx", 1, 0),
        ],

        # Layer 13
        [
            ("x", 0, theta_x3),
            ("x", 1, theta_x3),
        ],

        # Layer 14
        [
            ("cx", 0, 1),
        ],

        # Layer 15
        [
            ("y", 0, theta_y3),
            ("y", 1, theta_y3),
        ],

        # Layer 16
        [
            ("cx", 1, 0),
        ],

        # Layer 17
        [
            ("z", 0, theta_z3),
            ("z", 1, theta_z3),
        ],

        # Layer 18
        [
            ("cx", 0, 1),
        ],
    ],

    3: [
        # Layer 1
        [
            ("x", 0, theta_x1),
            ("x", 1, theta_x1),
            ("x", 2, theta_x1),
        ],

        # Layer 2
        [
            ("cx", 0, 1),
        ],

        # Layer 3
        [
            ("cx", 1, 2),
        ],

        # Layer 4
        [
            ("y", 0, theta_y1),
            ("y", 1, theta_y1),
            ("y", 2, theta_y1),
        ],

        # Layer 5
        [
            ("cx", 2, 1),
        ],

        # Layer 6
        [
            ("cx", 0, 1),
        ],

        # Layer 7
        [
            ("z", 0, theta_z1),
            ("z", 1, theta_z1),
            ("z", 2, theta_z1),
        ],

        # Layer 8
        [
            ("cx", 1, 2),
        ],

        # Layer 9
        [
            ("cx", 2, 1),
        ],

        # Layer 10
        [
            ("x", 0, theta_x2),
            ("x", 1, theta_x2),
            ("x", 2, theta_x2),
        ],

        # Layer 11
        [
            ("cx", 0, 1),
        ],

        # Layer 12
        [
            ("cx", 1, 2),
        ],

        # Layer 13
        [
            ("y", 0, theta_y2),
            ("y", 1, theta_y2),
            ("y", 2, theta_y2),
        ],

        # Layer 14
        [
            ("cx", 2, 1),
        ],

        # Layer 15
        [
            ("cx", 0, 1),
        ],

        # Layer 16
        [
            ("z", 0, theta_z2),
            ("z", 1, theta_z2),
            ("z", 2, theta_z2),
        ],

        # Layer 17
        [
            ("cx", 1, 2),
        ],

        # Layer 18
        [
            ("cx", 2, 1),
        ],
    ],

    4: [
        # Layer 1
        [
            ("x", 0, theta_x1),
            ("x", 1, theta_x1),
            ("x", 2, theta_x1),
            ("x", 3, theta_x1),
        ],

        # Layer 2
        [
            ("cx", 0, 1),
            ("cx", 2, 3),
        ],

        # Layer 3
        [
            ("y", 0, theta_y1),
            ("y", 1, theta_y1),
            ("y", 2, theta_y1),
            ("y", 3, theta_y1),
        ],

        # Layer 4
        [
            ("cx", 1, 2),
        ],

        # Layer 5
        [
            ("z", 0, theta_z1),
            ("z", 1, theta_z1),
            ("z", 2, theta_z1),
            ("z", 3, theta_z1),
        ],

        # Layer 6
        [
            ("cx", 3, 2),
            ("cx", 1, 0),
        ],

        # Layer 7
        [
            ("x", 0, theta_x2),
            ("x", 1, theta_x2),
            ("x", 2, theta_x2),
            ("x", 3, theta_x2),
        ],

        # Layer 8
        [
            ("cx", 2, 1),
        ],

        # Layer 9
        [
            ("y", 0, theta_y2),
            ("y", 1, theta_y2),
            ("y", 2, theta_y2),
            ("y", 3, theta_y2),
        ],

        # Layer 10
        [
            ("cx", 0, 1),
            ("cx", 2, 3),
        ],

        # Layer 11
        [
            ("z", 0, theta_z2),
            ("z", 1, theta_z2),
            ("z", 2, theta_z2),
            ("z", 3, theta_z2),
        ],

        # Layer 12
        [
            ("cx", 1, 2),
        ],

        # Layer 13
        [
            ("x", 0, theta_x3),
            ("x", 1, theta_x3),
            ("x", 2, theta_x3),
            ("x", 3, theta_x3),
        ],

        # Layer 14
        [
            ("cx", 3, 2),
            ("cx", 1, 0),
        ],

        # Layer 15
        [
            ("y", 0, theta_y3),
            ("y", 1, theta_y3),
            ("y", 2, theta_y3),
            ("y", 3, theta_y3),
        ],

        # Layer 16
        [
            ("cx", 2, 1),
        ],
    ],
}