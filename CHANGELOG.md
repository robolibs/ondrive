# Changelog

## [0.0.4] - 2026-11-19

### <!-- 0 -->⛰️  Features

- Plan forward-only tight turns, recheck at stops
- Drive into tight bends before turning
- Drive tight turns planned by turners
- Take the turners over from maptrax
- Steer at rest and reverse only as needed
- Drive tight turns in full-lock legs
- Pass movers behind, back off at cusps
- Plan reversing ways round tight turns
- Optimise a fresh band along the way too
- Take the shared, committed way past obstacles
- Obstacle predictions of a world
- Kalman obstacle tracker
- Post fields in tick_profile
- Thread guides through fields of obstacles
- Yield to crossing traffic
- Pre-steer for the leg past a cusp
- Reach poses on Ackermann by curves
- Add Frenet-frame trajectory follower
- Add velocity-obstacle path follower
- Separate guide hysteresis for traffic
- Model steering lag
- Followers avoid traffic and give way
- Steer path followers round obstacles
- Plan a dubins path to a bare goal
- Back away from head-on traffic
- Sample full acceleration to the ref speed
- Choose swerve and pace together
- Swerve behind crossing traffic
- Low-pass the commanded turn rate
- Soc samples smooth by default
- Smooth sampling default on in mppi
- Safety filter on for non-avoiding kinds
- Cbf safety filter, slip tracker, smppi
- Teb control lookahead parameter
- Teb gives way to crossing traffic
- Stopping distance in planning horizon
- Guided mppi sample, give-way solves
- Machine-scaled horizons, yielding
- Avoidance, actuation models, debug frames

### <!-- 1 -->🐛 Bug Fixes

- Reverse only until a forward leg ends it
- Creep past a drive's dead band
- Grade the soft risk across the margin
- Pursue on the commanded steering again
- Drive on past posts instead of waiting
- Detours keep past the MPPI margin
- Solve a way from a drive along it
- Threaded ways keep past the MPPI margin
- Solve a way from a drive along it
- Slower band at nominal timing, check it all
- Carry the chosen way into every pace
- Predict still obstacles to stay put
- Commit to a way, share it with MPC
- Bias against the steering reached
- Pick a detour side per obstacle cluster
- Track as tightly as mppi
- Tighten cross-track input range
- Track as tightly as mppi
- Hand the path to its curves
- Keep steering-rate cap to point kinds
- Round corners for point controllers
- Repel by footprint clearance
- Ease point carrots back after detours
- Follow the path it is handed
- Weigh path deviation three times more
- Hold cusps from where the jump went past
- Set carrots ahead of the predicted pose
- Cap speed to stop at each cusp
- Come at the goal along a run-in
- Integrate commands from the held speed
- Feed the bend's steering forward in reverse
- Read reverse errors at the rear axle
- Approach cusps on the arriving leg
- Stop at goals and path ends
- Close in inside the tolerance
- Aim past an open path's end
- Lead point controllers to a goal's heading
- Swing the pose out with the turn ahead
- Plan the band at the axle the body turns about
- Keep wider room round obstacles
- Make the potential field give way
- Keep the band model, brake early for cusps
- Pass carrots at speed, yield in step
- Aim past an open path's end
- Plan goals with steering in hand
- Give way for point controllers blind to traffic
- Stop at cusps and model the rear axle
- Clamp bends to what the machine can turn
- Reverse along cusp legs and stop at cusps
- Pass carrots without looping round
- Face stop goals the way the body drives
- Pass carrots at the path's speed
- Track bends by the pose and brake early
- Predict the pose's drift off its heading
- Brake ahead of the bends it plans
- Keep wider room from obstacles
- Commit to one side round obstacles
- Stop once well past an open path's end
- Unwind full lock when reversing
- Hold sideslip when lateral is unmeasured
- Hold the forward leg until its cusp
- Hold course through slip and slopes
- Steer by the rear axle
- Plan for the rear axle both ways
- Pass cusps closer, along the path too
- Keep the rear axle on the pose's leg
- Arrive slow enough to stop in time
- Aim to put the pose on the path
- Round corners for Ackermann followers
- Pursuits bring the pose, not the axle
- Pursuit lookahead shrinks for bends ahead
- Restart loops half way round a lap
- Slow by the path left, not the goal
- Smooth paths on a spline, pass cusps
- Followers reverse out of cusps
- Give way wider to crossing traffic
- Judge arrival at the robot pose
- Brake within the platform's deceleration
- Keep room from standing obstacles on wide bodies
- Integrate speed from the last command
- Keep room beside wide bodies on detours
- Track detour guides instead of braking late
- Leave crossing traffic to apf's own field
- Slow for the sharpest curve within braking
- Look 12 s ahead for moving traffic
- Limit turn-rate change by angular accel
- Preview curvature one steering delay ahead
- Brake to the next cusp, cap the band there
- Grow obstacle margin with steering lag
- Widen obstacle margin to 0.6 m
- Widen safety margin to 0.6 m
- Keep a 1 m obstacle margin
- Give way along the plan it drives
- Default smooth sampling on when unset
- Stand while in contact when giving way
- Mca hard risk on the bare footprint
- Car-like ends within its turning circle
- Teb faces the body heading on reverse legs
- Mppi keeps a safety margin round obstacles
- Soc centres samples on guide, pursues path
- Teb limits turn-rate change to steering
- Soc gives way, dwa caps speed for crossers
- Give way to crossers by keep-or-stand check
- Ilqr damps turn rate against path curvature
- Soc follows its detour guide when cheapest
- Mppi, mca, soc cap speed near reference
- Dwa cruises near its target speed
- Break stalls, cap mpc/ilqr speed
- Measured steering, no creep stall, soc detour
- Teb keeps 0.4 m from obstacles
- Rpp watches the held arc to the horizon
- Rpp detours with 0.2 m more room
- Mca keeps 0.3 m round its body
- Mppi gives way to crossing traffic
- Ackermann drives through the path end
- Point controllers finish past the end
- Only the path end completes a path
- Dwa yields to crossers until clear
- Soc correlated noise, guide hysteresis
- Paced give-way, stronger barrier
- Detours, pursue guide, crossing check
- Guards, path progress, reset, limits
- Propagate stage costs through dynamics
- Wrap reeds-shepp angles to [-pi, pi]

### <!-- 2 -->🚜 Refactor

- Every obstacle in the cost, no give-way

### <!-- 3 -->📚 Documentation

- Put the obstacle penalty doc on its fn
- Describe the reverse error blocks
- Horizons, yielding, controller table
- Obstacle handling, debug frames, make targets
- Point links at github

### <!-- 4 -->⚡ Performance

- Cost only the parts a nudge moves
- Sort standing obstacles once per checker
- Frame only the nearest projected segment
- Thread the guide once a tick
- Prune detours by obstacle reach per block
- Cap critic clearance, share crossings
- Look up obstacles near a pose in grid cells
- Leave out obstacles out of reach
- Thread pool, resumed rollouts, lto
- Cache predictions per step
- Share the guides' gradients among cores
- Drop idle gradients, fewer clear guides

### <!-- 6 -->🧪 Testing

- Give way to crossing traffic, not steer

### <!-- 7 -->⚙️ Miscellaneous Tasks

- Print the final pose in full
- Python binding test recipe

## [0.0.3] - 2026-10-09

## [0.0.3] - 2026-10-09

### <!-- 0 -->⛰️  Features

- Strafing for mpc/ilqr/mppi/mca/soc/teb
- True holonomic kinematics for point and path controllers
- Obstacle model, ilc, apf, adaptive lookahead
- Pose regulator, vector pursuit, kanayama, timed trajectories
- Regulated pursuit, pose reach, ilqr
- Rewrite controllers to canonical algorithms

### <!-- 3 -->📚 Documentation

- Point readme at controller roadmap

### <!-- 7 -->⚙️ Miscellaneous Tasks

- Migrate to oslo config, clean up layout

## [0.0.2] - 2026-06-06

### <!-- 0 -->⛰️  Features

- Readme.md
- CPP-PARITY

### Build

- Use codeberg remote for datapod

