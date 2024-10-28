import pandas as pd
from datetime import datetime, timedelta
import random

def generate_normal_audit_logs(num_sequences=50):
    """
    Generate synthetic audit logs with enhanced variety and realistic timing
    """
    # Define expanded activities with multiple variations
    activity_variations = {
        'login': [
            'User logged into the system',
            'Started shift with system login',
            'Initiated system access',
            'Completed authentication process',
            'Logged in for scheduled shift'
        ],
        'batch_prep': [
            'Loaded recipe for batch #{}',
            'Initialized production recipe #{}',
            'Prepared manufacturing formula for batch #{}',
            'Set up batch #{} parameters',
            'Configured system for batch #{}'
        ],
        'equipment_check': [
            'Initiated pre-operation equipment check',
            'Performed equipment validation check',
            'Completed machinery safety inspection',
            'Conducted standard equipment verification',
            'Executed pre-batch equipment diagnostic'
        ],
        'calibration': [
            'Verified calibration of scales',
            'Performed sensor calibration check',
            'Validated measurement systems',
            'Completed instrument calibration',
            'Checked and verified all gauges',
            'Calibrated pressure sensors',
            'Verified temperature probes'
        ],
        'batch_start': [
            'Started coating process for batch #{}',
            'Initiated production of batch #{}',
            'Began manufacturing sequence for #{}',
            'Commenced batch #{} processing',
            'Launched production run #{}'
        ],
        'process_monitoring': [
            'Monitored coating uniformity',
            'Checked product consistency',
            'Verified process parameters',
            'Assessed coating thickness',
            'Evaluated product quality metrics'
        ],
        'temp_adjust': [
            'Adjusted inlet air temperature from {}°C to {}°C',
            'Modified process temperature {}°C to {}°C',
            'Regulated air temperature: {}°C to {}°C',
            'Adjusted heating parameters {}°C to {}°C',
            'Fine-tuned temperature from {}°C to {}°C'
        ],
        'spray_adjust': [
            'Adjusted spray rate from {} mL/min to {} mL/min',
            'Modified coating flow rate {} to {} mL/min',
            'Changed spray parameters {} to {} mL/min',
            'Updated liquid flow rate {} to {} mL/min',
            'Regulated spray speed {} to {} mL/min'
        ],
        'environmental_check': [
            'Checked humidity levels',
            'Monitored room conditions',
            'Verified environmental parameters',
            'Assessed ambient conditions',
            'Recorded environmental metrics'
        ],
        'quality_check': [
            'Performed intermediate quality check',
            'Conducted in-process testing',
            'Executed quality verification',
            'Completed quality assessment',
            'Performed product inspection'
        ],
        'drum_speed': [
            'Adjusted drum speed from {} RPM to {} RPM',
            'Modified rotation rate {} to {} RPM',
            'Changed drum velocity {} to {} RPM',
            'Updated rotation speed {} to {} RPM',
            'Regulated drum RPM {} to {}'
        ],
        'solution_change': [
            'Changed coating solution to type {}',
            'Switched to solution variant {}',
            'Modified coating material to type {}',
            'Updated coating compound to {}',
            'Transitioned to solution {}'
        ],
        'documentation': [
            'Updated batch records',
            'Documented process parameters',
            'Recorded manufacturing data',
            'Completed batch documentation',
            'Updated electronic batch record'
        ],
        'maintenance': [
            'Performed routine equipment cleaning',
            'Conducted scheduled maintenance',
            'Completed equipment sanitization',
            'Executed standard cleaning procedure',
            'Performed equipment maintenance check'
        ],
        'batch_end': [
            'Stopped coating process for batch #{}',
            'Completed production of batch #{}',
            'Finalized batch #{} processing',
            'Concluded manufacturing of batch #{}',
            'Ended production sequence #{}'
        ],
        'logout': [
            'Logged out of system',
            'Completed system logout',
            'Ended user session',
            'Finished shift and logged out',
            'Terminated system access'
        ]
    }

    # Define normal parameter ranges
    temp_range = (145, 155)
    spray_rate_range = (8, 15)
    drum_speed_range = (12, 19)
    solution_types = ['A', 'B', 'C', 'D', 'E']
    users = ['user123', 'user456', 'user789', 'Sarah Johnson', 'Mark Wilson', 'Emily Davis',
             'Chris Wilson', 'Emily Chen', 'David Kim', 'Michael Brown', 'Lisa Anderson']

    # Define variable time intervals for different activities
    time_intervals = {
        'login': (2, 5),
        'batch_prep': (10, 20),
        'equipment_check': (15, 25),
        'calibration': (10, 15),
        'batch_start': (5, 10),
        'process_monitoring': (5, 15),
        'temp_adjust': (3, 8),
        'spray_adjust': (3, 8),
        'environmental_check': (5, 10),
        'quality_check': (15, 30),
        'drum_speed': (3, 8),
        'solution_change': (10, 20),
        'documentation': (5, 15),
        'maintenance': (20, 40),
        'batch_end': (10, 15),
        'logout': (2, 5)
    }

    synthetic_logs = []

    # Generate sequences
    for seq in range(num_sequences):
        # Set up sequence parameters
        current_date = datetime.strptime('2024-08-27', '%Y-%m-%d') + timedelta(days=seq)
        current_time = datetime.strptime('08:00:00', '%H:%M:%S')
        current_user = random.choice(users)
        batch_number = f"{seq+1:03d}"

        # Base sequence of activities
        base_sequence = ['login', 'batch_prep', 'equipment_check', 'calibration', 'batch_start']
        
        # Add random process monitoring and adjustments
        middle_activities = ['process_monitoring', 'temp_adjust', 'spray_adjust', 'environmental_check', 
                           'quality_check', 'drum_speed', 'solution_change', 'documentation']
        
        # Shuffle middle activities and select a random number of them
        random.shuffle(middle_activities)
        selected_middle = middle_activities[:random.randint(5, len(middle_activities))]
        
        # End sequence
        end_sequence = ['maintenance', 'batch_end', 'logout']
        
        # Combine all activities
        full_sequence = base_sequence + selected_middle + end_sequence

        # Generate the sequence
        for activity_type in full_sequence:
            if activity_type in ['batch_prep', 'batch_start', 'batch_end']:
                activity = random.choice(activity_variations[activity_type]).format(batch_number)
            elif activity_type == 'temp_adjust':
                temp1 = random.randint(*temp_range)
                temp2 = min(max(temp1 + random.randint(-2, 2), temp_range[0]), temp_range[1])
                activity = random.choice(activity_variations[activity_type]).format(temp1, temp2)
            elif activity_type == 'spray_adjust':
                rate1 = random.randint(*spray_rate_range)
                rate2 = min(max(rate1 + random.randint(-2, 2), spray_rate_range[0]), spray_rate_range[1])
                activity = random.choice(activity_variations[activity_type]).format(rate1, rate2)
            elif activity_type == 'drum_speed':
                speed1 = random.randint(*drum_speed_range)
                speed2 = min(max(speed1 + random.randint(-2, 2), drum_speed_range[0]), drum_speed_range[1])
                activity = random.choice(activity_variations[activity_type]).format(speed1, speed2)
            elif activity_type == 'solution_change':
                activity = random.choice(activity_variations[activity_type]).format(random.choice(solution_types))
            else:
                activity = random.choice(activity_variations[activity_type])

            # Add log entry
            synthetic_logs.append([
                current_date.strftime('%Y-%m-%d'),
                current_time.strftime('%H:%M:%S'),
                current_user,
                activity,
                'Not Available',
                '0'
            ])

            # Add realistic time interval based on activity type
            min_time, max_time = time_intervals[activity_type]
            current_time += timedelta(minutes=random.randint(min_time, max_time))

    return pd.DataFrame(synthetic_logs, columns=['Date', 'Time', 'User', 
                                               'Activity Description', 'Reason for change', 'Anomaly'])

def save_audit_logs(num_sequences=1500, output_file='enhanced_audit_logs.csv'):
    """
    Generate and save enhanced audit logs
    """
    try:
        df = generate_normal_audit_logs(num_sequences)
        df.to_csv(output_file, index=False)
        print(f"Successfully saved {len(df)} records to {output_file}")
        print(f"Generated {num_sequences} complete sequences")
        return True
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return False

# Run this code directly
if __name__ == "__main__":
    save_audit_logs(num_sequences=1500, output_file='enhanced_audit_logs1.5k.csv')