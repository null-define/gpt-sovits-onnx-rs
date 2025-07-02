use std::fs;
use std::io;

pub fn get_hw_cpu_ids_and_freqs() -> io::Result<Vec<(usize, u64)>> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let mut cpu_info: Vec<(usize, u64)> = Vec::new();
        let mut seen_siblings: Vec<String> = Vec::new();
        let cpu_dir = "/sys/devices/system/cpu/";

        for entry in fs::read_dir(cpu_dir)? {
            let entry = entry?;
            let path = entry.path();
            let file_name = path.file_name().unwrap().to_str().unwrap();
            if path.is_dir()
                && file_name.starts_with("cpu")
                && file_name.chars().skip(3).all(|c| c.is_digit(10))
            {
                let cpu_id: usize = file_name[3..].parse().unwrap();

                // Read thread siblings to filter hyper-threaded CPUs
                let siblings_path = path.join("topology/thread_siblings_list");
                let siblings = fs::read_to_string(siblings_path).unwrap_or(cpu_id.to_string());

                // Skip if this physical core was already processed
                if seen_siblings.contains(&siblings) {
                    continue;
                }
                seen_siblings.push(siblings);

                // Read max frequency
                let freq_path = path.join("cpufreq/cpuinfo_max_freq");
                let freq = fs::read_to_string(freq_path)
                    .map(|s| s.trim().parse::<u64>().unwrap_or(0))
                    .unwrap_or(0);

                cpu_info.push((cpu_id, freq));
            }
        }

        // Sort by frequency (descending) and CPU ID (ascending)
        cpu_info.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        Ok(cpu_info)
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        use std::thread;
        // On non-Linux platforms, return CPU IDs in natural order with frequency 0
        let num_cpus = thread::available_parallelism()?.get();
        Ok((0..num_cpus).map(|id| (id, 0)).collect())
    }
}

pub fn get_hw_big_cores() -> io::Result<Vec<(usize, u64)>> {
    let cpu_info = get_hw_cpu_ids_and_freqs()?;

    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        if cpu_info.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "No CPU information available",
            ));
        }
        // if cpu cores less than 4, return all
        if cpu_info.len() < 4 {
            return Ok(cpu_info);
        }

        // Find the frequency threshold for big cores (e.g., highest frequency group)
        let big_core_freq = cpu_info[3].1;
        if big_core_freq == 0 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "Frequency information unavailable",
            ));
        }

        // Collect cores with high frequency (assume big cores have the highest frequency)
        let big_cores: Vec<(usize, u64)> = cpu_info
            .into_iter()
            .filter(|&(_, freq)| freq >= big_core_freq) // Big cores have the highest frequency
            .collect();
        Ok(big_cores)
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        // Non-Linux platforms: frequency is 0, cannot reliably identify big cores
        Err(io::Error::new(
            io::ErrorKind::Other,
            "Cannot identify big cores on non-Linux platforms due to missing frequency data",
        ))
    }
}
