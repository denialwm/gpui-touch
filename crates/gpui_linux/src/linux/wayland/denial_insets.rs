//! Denial's insets protocol, for the on-screen keyboard's height over a window
//! (https://github.com/denialwm/denial, compositor/protocol/denial-insets-v1.xml).

#[allow(
    dead_code,
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    unused_imports,
    unused_unsafe,
    unused_variables,
    clippy::all,
    missing_docs
)]
mod protocol {
    use wayland_client;
    use wayland_client::protocol::*;
    pub mod __interfaces {
        use wayland_client::backend as wayland_backend;
        use wayland_client::protocol::__interfaces::*;
        wayland_scanner::generate_interfaces!("protocols/denial-insets-v1.xml");
    }
    use self::__interfaces::*;
    wayland_scanner::generate_client_code!("protocols/denial-insets-v1.xml");
}

pub(crate) use protocol::{
    denial_insets_manager_v1::DenialInsetsManagerV1,
    denial_keyboard_inset_v1::{self, DenialKeyboardInsetV1},
};
